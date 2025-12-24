#  Natural Impair–Repair（NIR）+ 表示层锚定/去相似化 + KD + Anchor
#
#  目标：在保持 retain/test 精度的同时，增强 forget 的遗忘效果。
#  关键思想：
#   1) forget：SpecAugment 输入扰动 + KL 到 Uniform（高熵遗忘）
#   2) retain：CE 保精度 + KD 到 teacher（遗忘前快照）
#   3) Anchor：限制 adapter 权重漂移
#   4) 表示层约束：retain 表示对齐 / forget 表示去相似化

import os
import copy
import itertools
import random
import sys
import csv

from typing import Optional
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from utils import *
from ebranchformer import *
from data.process_data.extract_fbank_feature import *

BASE_MODEL_PATH = "./log1/base_clean_adapter3.pth"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "data", "all_data")
TRAIN_SPLIT_DIR = os.path.join(DATA_ROOT, "train_split_by_patient")
VAL_DIR = os.path.join(DATA_ROOT, "validation")
TEST_DIR = os.path.join(DATA_ROOT, "test")


def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _to_padded_batch(wavs, device):
    """将变长特征列表转换为（padding后的张量，长度）."""
    # wavs: List[np.ndarray]，每个元素是变长特征 [T, D]（fbank）或 [T]
    tensors = [torch.as_tensor(w, dtype=torch.float32, device=device) for w in wavs]
    lengths = torch.IntTensor([t.shape[0] for t in tensors]).to(device)
    padded = pad_sequence(tensors, batch_first=True)  # [B, T, D]
    return padded, lengths


def spec_augment_fbank(x, lengths, time_mask_param=20, freq_mask_param=10, num_time_masks=2, num_freq_masks=2):
    """轻量级“自然扰动”模块：对 fbank 做 SpecAugment（时间遮挡/频带遮挡）。

    仅在 forget-batch 上使用，用“输入层扰动”来替代不自然的标签篡改，
    让遗忘过程更接近 NatMU 提倡的 natural unlearning 思路。
    """
    if x.dim() != 3:
        return x
    x = x.clone()
    B, T, D = x.shape
    for b in range(B):
        L = int(lengths[b].item())
        if L <= 1:
            continue

        # 时间遮挡（Time Mask）
        for _ in range(num_time_masks):
            max_w = min(time_mask_param, max(1, L // 2))
            w = random.randint(0, max_w)
            if w <= 0:
                continue
            t0 = random.randint(0, max(0, L - w))
            x[b, t0:t0 + w, :] = 0.0

        # 频带遮挡（Frequency Mask）
        for _ in range(num_freq_masks):
            max_f = min(freq_mask_param, max(1, D // 2))
            f = random.randint(0, max_f)
            if f <= 0:
                continue
            f0 = random.randint(0, max(0, D - f))
            x[b, :, f0:f0 + f] = 0.0
    return x


def kd_kl(student_logits, teacher_logits, T=2.0):
    """带温度缩放的 KL 蒸馏损失（用于 retain 侧保持行为）。"""
    return F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction="batchmean",
    ) * (T * T)


def uniform_kl(logits, class_num: int, T=2.0):
    """KL(student || Uniform)：让 forget 样本输出更高熵/更低置信度（实现“自然遗忘”）。"""
    target_prob = torch.full_like(logits, 1.0 / class_num)
    return F.kl_div(
        F.log_softmax(logits / T, dim=1),
        target_prob,
        reduction="batchmean",
    ) * (T * T)


# ---------------- 表示层约束模块（可作为“我的贡献点”写进论文） ----------------
def _extract_tensor(out):
    """从 hook 输出中提取“表示张量”。

    说明：
    - 很多 encoder 层 forward 可能返回 (x, mask) 或 {"x": x, "mask": mask}
    - 这里优先选择 **浮点型** 且形状像 [B, T, D] 的张量
    - 自动跳过 bool / int 这类 mask，避免后续 mean() 报错
    """

    def _walk(obj):
        if torch.is_tensor(obj):
            yield obj
        elif isinstance(obj, dict):
            for v in obj.values():
                yield from _walk(v)
        elif isinstance(obj, (tuple, list)):
            for v in obj:
                yield from _walk(v)

    float_seq = None  # 优先：序列表示 [B, T, D] 等
    float_any = None  # 其次：任意浮点张量

    for t in _walk(out):
        if not torch.is_tensor(t):
            continue
        if t.dtype == torch.bool:
            continue
        if not (t.dtype.is_floating_point or t.dtype.is_complex):
            continue

        if float_any is None:
            float_any = t
        # 常见序列表示维度 >= 3（例如 [B, T, D]）
        if t.dim() >= 3:
            float_seq = t
            break

    return float_seq if float_seq is not None else float_any


def _masked_mean_time(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """将序列表示池化为 [B, D]（按长度 mask 做平均池化，避免 padding 干扰）。"""
    # 防御性处理：如果意外拿到 bool mask（而不是表示张量），先转为 float 避免 mean() 直接崩溃。
    # 正常情况下 _extract_tensor 会跳过 bool，因此这里更多是兜底。
    if torch.is_tensor(x) and x.dtype == torch.bool:
        x = x.float()
    if not torch.is_tensor(x) or x.dim() < 2:
        return x
    if lengths is None or lengths.numel() == 0:
        return x.mean(dim=1) if x.dim() == 3 else x

    # 常见形状：[B, T, D] 或 [B, D, T]
    B = x.shape[0]
    if x.dim() == 3:
        if x.shape[1] == lengths.max().item() or x.shape[1] >= lengths.max().item():
            T = x.shape[1]
            mask = (torch.arange(T, device=x.device)[None, :] < lengths[:, None]).float()
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            return (x * mask.unsqueeze(-1)).sum(dim=1) / denom
        if x.shape[2] >= lengths.max().item():
            T = x.shape[2]
            mask = (torch.arange(T, device=x.device)[None, :] < lengths[:, None]).float()
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            return (x * mask.unsqueeze(1)).sum(dim=2) / denom
        # 兜底策略
        return x.mean(dim=1)
    if x.dim() == 2:
        return x
    # 兜底策略：处理更高维张量的情况
    return x.reshape(B, -1)


def register_encoder_hooks(model, layer_indices):
    """在指定 encoder 层注册 forward hook，用于抓取中间层表示。"""
    if not hasattr(model, "encoders"):
        raise AttributeError("Model has no attribute 'encoders'. Please expose encoder blocks as model.encoders.")
    cache = {}
    hooks = []

    def _make_hook(key):
        def _hook(module, inputs, output):
            t = _extract_tensor(output)
            if t is None:
                # 有些层 output 不是 tensor/不含表示张量，则尝试从 inputs 中取
                t = _extract_tensor(inputs)
            cache[key] = t

        return _hook

    for idx in layer_indices:
        layer = model.encoders[idx]
        hooks.append(layer.register_forward_hook(_make_hook(idx)))
    return cache, hooks


def rep_mse_loss(student_cache, teacher_cache, lengths, layer_indices):
    """retain 侧表示锚定：选定层的池化表示做 MSE（让 student 表示贴近 teacher）。"""
    losses = []
    for idx in layer_indices:
        hs = student_cache.get(idx, None)
        ht = teacher_cache.get(idx, None)
        if hs is None or ht is None:
            continue
        vs = _masked_mean_time(hs, lengths)
        vt = _masked_mean_time(ht, lengths)
        losses.append(F.mse_loss(vs, vt))
    return sum(losses) / max(len(losses), 1)


def rep_cos2_loss(student_cache, teacher_cache, lengths, layer_indices):
    """forget 侧表示去相似化：最小化 cosine^2，使表示近似正交（稳定的 repulsion）。"""
    losses = []
    for idx in layer_indices:
        hs = student_cache.get(idx, None)
        ht = teacher_cache.get(idx, None)
        if hs is None or ht is None:
            continue
        vs = _masked_mean_time(hs, lengths)
        vt = _masked_mean_time(ht, lengths)
        cos = F.cosine_similarity(vs, vt, dim=1).clamp(-1.0, 1.0)
        losses.append((cos ** 2).mean())
    return sum(losses) / max(len(losses), 1)


def evaluate(model, dataloader, weights, class_num, device):
    """评估函数：修正 CE/Focal 的实现（CE 必须输入 logits，而不是 softmax 概率）。"""
    model.eval()
    losses = []
    true_labels = []
    predicted_labels = []
    weights = weights.to(device)

    with torch.no_grad():
        for _, (wavs, labels, _, _) in enumerate(dataloader):
            labels = convert_labels(labels, class_num)
            labels = torch.LongTensor(labels).to(device)

            features, features_len = _to_padded_batch(wavs, device)
            logits = model(features, features_len)  # [B, C]
            probs = F.softmax(logits, dim=1)

            # 预测类别
            _, pred_cls = torch.max(probs, dim=1)

            # CE 直接作用于 logits（关键：不是 softmax）
            ce = F.cross_entropy(logits, labels, reduction='none')  # [B]
            # Focal 加权（可选）
            pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1).clamp_min(1e-8)
            gamma = 1.0
            fl = weights[labels] * ((1 - pt) ** gamma) * ce
            loss = fl.mean()

            losses.append(loss.item())
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(pred_cls.cpu().numpy())

    accuracy = accuracy_score(np.array(true_labels), np.array(predicted_labels))
    macro_f1 = f1_score(np.array(true_labels), np.array(predicted_labels), average='macro')
    return float(np.mean(losses)), float(accuracy), float(macro_f1)


# ==============================================================
#  MIA-AUC（成员推断攻击）评估模块
#  目的：衡量“遗忘有效性”——遗忘后，forget 样本应更像 non-member（测试集），
#        攻击者越难区分 member/non-member 越好（AUC 越接近 0.5 越好）。
#
#  我们这里将：member = forget_dataloader（曾参与训练、请求删除）
#            non-member = test_dataloader（未参与训练）
#
#  打分方式（score_type）：
#    - 'neg_loss'：-CE_loss（成员通常loss更低→score更高，AUC越高越容易被攻击）
#    - 'max_conf'：max softmax 置信度（成员通常更自信→score更高）
# ==============================================================

def _collect_mia_scores(model, dataloader, class_num, device,
                        score_type: str = "neg_loss",
                        max_samples: Optional[int] = 2000):
    """收集用于 MIA 的样本打分（不改变训练）。"""
    model.eval()
    scores = []

    with torch.no_grad():
        for _, (wavs, labels, _, _) in enumerate(dataloader):
            labels = convert_labels(labels, class_num)
            labels = torch.LongTensor(labels).to(device)

            features, features_len = _to_padded_batch(wavs, device)
            logits = model(features, features_len)  # [B, C]
            probs = F.softmax(logits, dim=1)  # [B, C]

            if score_type == "neg_loss":
                ce = F.cross_entropy(logits, labels, reduction="none")  # [B]
                score = (-ce).detach().cpu().numpy()
            elif score_type == "max_conf":
                score = probs.max(dim=1).values.detach().cpu().numpy()
            else:
                raise ValueError(f"Unsupported score_type: {score_type}")

            scores.append(score)

            if max_samples is not None and sum(len(x) for x in scores) >= max_samples:
                break

    if len(scores) == 0:
        return np.array([], dtype=np.float32)

    scores = np.concatenate(scores, axis=0)
    if max_samples is not None and scores.shape[0] > max_samples:
        scores = scores[:max_samples]
    return scores.astype(np.float32)


def mia_auc(model,
            member_loader,
            nonmember_loader,
            class_num,
            device,
            score_type: str = "neg_loss",
            max_samples: Optional[int] = 2000,
            balance: bool = True) -> float:
    """计算 MIA 的 ROC-AUC。AUC 越接近 0.5 表示越“不可区分”（更符合遗忘目标）。"""
    s_m = _collect_mia_scores(model, member_loader, class_num, device, score_type, max_samples)
    s_n = _collect_mia_scores(model, nonmember_loader, class_num, device, score_type, max_samples)

    if s_m.size == 0 or s_n.size == 0:
        return float("nan")

    if balance:
        n = min(len(s_m), len(s_n))
        s_m = s_m[:n]
        s_n = s_n[:n]

    y = np.concatenate([np.ones_like(s_m, dtype=np.int32),
                        np.zeros_like(s_n, dtype=np.int32)], axis=0)
    s = np.concatenate([s_m, s_n], axis=0)

    # 若某种极端情况下只有一个类别，roc_auc_score 会报错，这里做兜底
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


# =========================
#  置信度统计 & JS散度模块
# =========================
def collect_max_conf(model,
                     dataloader,
                     class_num: int,
                     device,
                     max_samples: Optional[int] = 2000):
    """
    收集一批样本的“最大Softmax置信度”（max confidence，范围[0,1]）。
    用于：
      1) 画置信度直方图
      2) 计算 mean/max confidence
      3) 计算分布距离（JS散度）
    """
    model.eval()
    scores = []
    with torch.no_grad():
        for wavs, labels, _, _ in dataloader:
            feat, lengths = _to_padded_batch(wavs, device)
            logits = model(feat, lengths)  # [B, C]
            probs = F.softmax(logits, dim=-1)  # [B, C]
            mx = probs.max(dim=-1).values.detach().cpu().numpy().astype(np.float32)  # [B]
            scores.append(mx)
            if max_samples is not None:
                if sum(x.shape[0] for x in scores) >= max_samples:
                    break
    if len(scores) == 0:
        return np.zeros((0,), dtype=np.float32)
    scores = np.concatenate(scores, axis=0)
    if max_samples is not None and scores.shape[0] > max_samples:
        scores = scores[:max_samples]
    return scores


def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """计算 JS Divergence（对称，范围[0, ln2]；这里返回自然对数底）。"""
    p = p.astype(np.float64) + eps
    q = q.astype(np.float64) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def js_divergence_from_scores(a_scores: np.ndarray,
                              b_scores: np.ndarray,
                              bins: int = 10) -> float:
    """
    将置信度分数映射到直方图分布后，计算 JS散度。
    bins=10 对应 0-0.1,0.1-0.2,...,0.9-1.0（与你柱状图区间一致）。
    """
    if a_scores.size == 0 or b_scores.size == 0:
        return float("nan")
    a_hist, _ = np.histogram(a_scores, bins=bins, range=(0.0, 1.0))
    b_hist, _ = np.histogram(b_scores, bins=bins, range=(0.0, 1.0))
    return _js_divergence(a_hist, b_hist)


def confidence_table(retain_conf: np.ndarray, forget_conf: np.ndarray, test_conf: np.ndarray):
    """返回一个用于打印的(均值, 最大值)字典。"""

    def _stat(x):
        if x.size == 0:
            return float("nan"), float("nan")
        return float(x.mean()), float(x.max())

    r_mean, r_max = _stat(retain_conf)
    f_mean, f_max = _stat(forget_conf)
    t_mean, t_max = _stat(test_conf)

    return {
        "Retain": (r_mean, r_max),
        "Forget": (f_mean, f_max),
        "Test": (t_mean, t_max),
    }


def print_conf_table(title: str, tab: dict, js_ft: float):
    """以“论文可用”的简洁格式打印置信度表 + JS散度。"""
    print(f"[{title}] mean/max confidence（范围0~1） & JS(Forget||Test)={js_ft:.4f}")
    print("    集合    mean_conf    max_conf")
    for k in ["Retain", "Forget", "Test"]:
        mean_c, max_c = tab[k]
        print(f"    {k:<6}  {mean_c:>9.4f}   {max_c:>8.4f}")


def save_metrics_history_csv(metrics_csv: str, metrics_history: list):
    """将每个 epoch 的指标写入 CSV（便于论文制表/画图）。"""
    out_dir = os.path.dirname(metrics_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 统一列顺序（若某列不存在则留空）
    columns = [
        "epoch",
        "retain_acc", "retain_f1",
        "forget_acc", "forget_f1",
        "test_acc", "test_f1",
        "s",
        "mia_auc",
        "retain_mean_conf", "retain_max_conf",
        "forget_mean_conf", "forget_max_conf",
        "test_mean_conf", "test_max_conf",
        "js_forget_test",
    ]

    import csv
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for row in metrics_history:
            w.writerow([row.get(c, "") for c in columns])


def plot_metrics_curves(metrics_png: str, metrics_history: list):
    """保存一张“曲线图PNG”：Acc/MIA-AUC（左轴）+ JS(Forget||Test)（右轴）。"""
    # 只取 epoch 为数字的记录（跳过 final_bestmia 这类字符串）
    xs = []
    retain_accs, forget_accs, test_accs = [], [], []
    mia_aucs, js_vals = [], []
    for row in metrics_history:
        ep = row.get("epoch", None)
        if isinstance(ep, int) or (isinstance(ep, float) and ep.is_integer()):
            ep = int(ep)
        else:
            continue
        xs.append(ep)
        retain_accs.append(row.get("retain_acc", None))
        forget_accs.append(row.get("forget_acc", None))
        test_accs.append(row.get("test_acc", None))
        mia_aucs.append(row.get("mia_auc", None))
        js_vals.append(row.get("js_forget_test", None))

    if len(xs) == 0:
        return

    # 画图（不指定颜色，让 matplotlib 使用默认配色）
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.plot(xs, retain_accs, label="retain_acc")
    ax1.plot(xs, forget_accs, label="forget_acc")
    ax1.plot(xs, test_accs, label="test_acc")
    ax1.plot(xs, mia_aucs, label="mia_auc")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("acc / mia_auc")
    ax1.set_ylim(0.0, 1.0)

    ax2 = ax1.twinx()
    ax2.plot(xs, js_vals, label="JS(Forget||Test)", linestyle="--")
    ax2.set_ylabel("JS(Forget||Test)")

    # 合并图例
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")

    fig.tight_layout()
    out_dir = os.path.dirname(metrics_png)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(metrics_png, dpi=200)
    plt.close(fig)


def blindspot_unlearner_nir(
        model,
        teacher_model,
        train_forget_dataloader,
        train_retain_dataloader,
        forget_dataloader,
        retain_dataloader,
        test_dataloader,
        epochs,
        optimizer,
        weights,
        class_num,
        bestsavename,
        lastname,
        device,
        # --- 算法组件开关（不是“调参”，而是定义方法结构的模块权重） ---
        T_forget=5.0,
        T_kd=2.0,
        lam_forget=1.0,
        lam_ng=1.0,  # NegGrad（-CE反学习）权重：增强遗忘
        k_forget=3,  # Impair-first 日程：连续k步遗忘，再1步修复
        retain_floor=0.80,  # ✅ 只要保留集精度 >= 该阈值，就允许继续加强遗忘/并参与best_mia选择
        adaptive_k_forget=True,  # ✅ 是否按保留精度自动调整k_forget（更像方法而不是手调）
        k_forget_max=6,  # k_forget 最大值（遗忘步占比上限）
        # --- 加速遗忘：让遗忘效果更早出现（不是盲目加epoch） ---
        forget_warmup_epochs=5,  # 前N个epoch更偏向遗忘（更快出现遗忘趋势）
        k_forget_warmup=6,  # warmup阶段使用的k_forget（>=k_forget）
        forget_repeat=2,  # 每个epoch重复遍历forget数据的次数（等价于增加遗忘更新步数）
        target_mia=0.62,  # 达到该MIA-AUC且retain>=retain_floor时可提前停止
        retain_patience=2,  # 连续多少个epoch低于retain_floor就提前停止（防止精度崩）
        lam_ce=0.3,
        lam_kd=0.5,
        lam_anchor=1e-4,
        rep_layer_indices=None,
        lam_rep_retain=0.2,
        lam_rep_forget=0.05,
):
    """Natural Impair–Repair（NIR）+ 表示层约束 的 Adapter 机器遗忘框架。

    Impair（遗忘分支，forget）：
      - 对输入做 SpecAugment 风格扰动（更“自然”）
      - 对输出做 KL 到 Uniform（拉高熵/压低置信度）
      - NegGrad：对 forget 样本做 -CE（梯度上升），直接削弱判别能力

    训练日程（Impair-first schedule）：
      - 连续 k_forget 步只做遗忘（Impair），再 1 步做修复（Repair），缓解梯度冲突

    Repair（修复分支，retain）：
      - retain 上做标准交叉熵 CE（保精度）
      - retain 上做 KD 蒸馏到 teacher（teacher=遗忘前快照，保行为）

    Anchor（参数锚定）：
      - 对可训练参数（adapter）加 L2 漂移惩罚，限制更新幅度

    表示层约束（Representation constraints）：
      - retain 表示锚定：中间层/最后层表示对齐 teacher（MSE）
      - forget 表示去相似化：最后层表示与 teacher 表示近似正交（cos^2）

    这一套组合用于：在“强遗忘”的同时尽量保持 retain/test 精度。
    """
    # --- 记录每个epoch的MIA-AUC/置信度/JS散度，用于画曲线和写表格 ---
    metrics_history = []  # 每条: dict(epoch=..., retain_acc=..., mia_auc=..., js_ft=..., ...)
    metrics_csv = bestsavename.replace('.pth', '_metrics.csv')
    metrics_png = bestsavename.replace('.pth', '_metrics.png')

    result_array = np.empty((0, 3))

    # --- 保存可训练参数初值，用于 Anchor 正则（控制漂移） ---
    init_params = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    # --- 表示层 hook（默认：中间层 + 最后一层 encoder） ---
    if rep_layer_indices is None:
        if hasattr(model, "encoders"):
            n_layers = len(model.encoders)
            rep_layer_indices = sorted(set([max(0, n_layers // 2), n_layers - 1]))
        else:
            rep_layer_indices = []
    rep_layer_indices = list(rep_layer_indices) if rep_layer_indices else []

    if rep_layer_indices:
        s_cache, s_hooks = register_encoder_hooks(model, rep_layer_indices)
        t_cache, t_hooks = register_encoder_hooks(teacher_model, rep_layer_indices)
        # forget 侧 repulsion 只用最后一层（最任务相关、效果最直接）
        rep_layers_forget = [rep_layer_indices[-1]]
    else:
        s_cache, t_cache, s_hooks, t_hooks, rep_layers_forget = {}, {}, [], [], []
    # --- 遗忘前评估（Before） ---
    _, acc_retain, f1_retain = evaluate(model, retain_dataloader, weights, class_num, device)
    _, acc_forget, f1_forget = evaluate(model, forget_dataloader, weights, class_num, device)
    _, acc_test, f1_test = evaluate(model, test_dataloader, weights, class_num, device)
    mia_before = mia_auc(model, forget_dataloader, test_dataloader, class_num, device,
                         score_type='neg_loss', max_samples=2000, balance=True)
    # AUC 越接近 0.5 越“不可区分”（越符合遗忘/隐私目标）

    # -------- 置信度统计（Before）& JS散度（Forget vs Test）--------
    retain_conf_b = collect_max_conf(model, retain_dataloader, class_num, device, max_samples=2000)
    forget_conf_b = collect_max_conf(model, forget_dataloader, class_num, device, max_samples=2000)
    test_conf_b = collect_max_conf(model, test_dataloader, class_num, device, max_samples=2000)
    js_ft_before = js_divergence_from_scores(forget_conf_b, test_conf_b, bins=10)
    tab_b = confidence_table(retain_conf_b, forget_conf_b, test_conf_b)
    print_conf_table("Before-Metrics", tab_b, js_ft_before)

    # 记录遗忘前（Before）的指标：用于论文做 Before vs After 对比
    metrics_history.append({
        "epoch": 0,
        "retain_acc": float(acc_retain),
        "retain_f1": float(f1_retain),
        "forget_acc": float(acc_forget),
        "forget_f1": float(f1_forget),
        "test_acc": float(acc_test),
        "test_f1": float(f1_test),
        # Before 阶段还没有 s（综合指标），用空值占位
        "s": "",
        "mia_auc": float(mia_before),
        "retain_mean_conf": tab_b["Retain"][0],
        "retain_max_conf": tab_b["Retain"][1],
        "forget_mean_conf": tab_b["Forget"][0],
        "forget_max_conf": tab_b["Forget"][1],
        "test_mean_conf": tab_b["Test"][0],
        "test_max_conf": tab_b["Test"][1],
        "js_forget_test": float(js_ft_before),
    })
    print(f"[Before] retain_acc={acc_retain:.4f}, retain_f1={f1_retain:.4f}, "
          f"forget_acc={acc_forget:.4f}, forget_f1={f1_forget:.4f}, "
          f"test_acc={acc_test:.4f}, test_f1={f1_test:.4f}, mia_auc={mia_before:.4f}")
    result_array = np.vstack((result_array, [acc_retain, acc_forget, acc_test]))
    before_retain = acc_retain
    before_forget = acc_forget
    best_s = -1e9

    # --- 以“保留精度约束下的最小 MIA-AUC”为目标保存最佳模型（更符合机器遗忘论文口径） ---
    best_mia = 1e9
    bestmia_savename = bestsavename.replace('.pth', '_bestmia.pth')
    bestmia_epoch = -1
    retain_bad_epochs = 0

    try:
        # 让 retain loader 循环供给（每个 forget batch 后拿一批 retain 做修复）
        retain_iter = itertools.cycle(train_retain_dataloader)
        global_step = 0  # 贯穿所有epoch的步计数，用于Impair-first日程

        for epoch in range(epochs):
            model.train()

            # warmup：前 forget_warmup_epochs 个epoch提高遗忘步占比，让遗忘效果更早出现
            k_forget_cur = k_forget_warmup if epoch < forget_warmup_epochs else k_forget
            # 在每个epoch内可重复遍历forget数据，等价于增加遗忘更新步数（更快出现遗忘趋势）
            for _rep in range(forget_repeat):
                for (wavs_f, labels_f, _, _) in train_forget_dataloader:
                    global_step += 1
                    # Impair-first 日程：连续 k_forget 步只做遗忘，再 1 步做修复
                    do_repair = (global_step % (k_forget_cur + 1) == 0)

                    # ====== forget 分支（Impair：自然遗忘）======
                    feat_f, len_f = _to_padded_batch(wavs_f, device)
                    feat_f_aug = spec_augment_fbank(feat_f, len_f)  # 自然输入扰动（SpecAugment）
                    if rep_layer_indices:
                        s_cache.clear()
                    logits_f = model(feat_f_aug, len_f)

                    # NegGrad：对 forget 做“反学习”（梯度上升），直接削弱判别能力
                    labels_f_ng = convert_labels(labels_f, class_num)
                    labels_f_ng = torch.LongTensor(labels_f_ng).to(device)
                    loss_ng = -F.cross_entropy(logits_f, labels_f_ng)

                    # 高熵遗忘：输出拉向均匀分布（压低置信度）
                    loss_uniform = uniform_kl(logits_f, class_num=class_num, T=T_forget)

                    # 合并：结构融合（不是调参）
                    loss_forget = loss_uniform + lam_ng * loss_ng

                    # forget 表示去相似化（只用最后一层）
                    loss_rep_forget = torch.tensor(0.0, device=device)
                    if rep_layers_forget and lam_rep_forget > 0:
                        t_cache.clear()
                        with torch.no_grad():
                            _ = teacher_model(feat_f, len_f)
                        loss_rep_forget = rep_cos2_loss(s_cache, t_cache, len_f, rep_layers_forget)

                    # ====== Anchor：adapter 参数漂移控制（每一步都做，防止更新过猛）======
                    anchor = 0.0
                    for n, p in model.named_parameters():
                        if p.requires_grad:
                            anchor = anchor + (p - init_params[n]).pow(2).mean()

                    # ====== Repair 分支（只在 do_repair=True 时执行）======
                    loss_ce = torch.tensor(0.0, device=device)
                    loss_kd = torch.tensor(0.0, device=device)
                    loss_rep_retain = torch.tensor(0.0, device=device)

                    if do_repair:
                        wavs_r, labels_r, _, _ = next(retain_iter)
                        labels_r = convert_labels(labels_r, class_num)
                        labels_r = torch.LongTensor(labels_r).to(device)

                        feat_r, len_r = _to_padded_batch(wavs_r, device)
                        if rep_layer_indices:
                            s_cache.clear()
                            t_cache.clear()
                        logits_r = model(feat_r, len_r)
                        loss_ce = F.cross_entropy(logits_r, labels_r)

                        with torch.no_grad():
                            t_logits = teacher_model(feat_r, len_r)
                        loss_kd = kd_kl(logits_r, t_logits, T=T_kd)

                        if rep_layer_indices and lam_rep_retain > 0:
                            loss_rep_retain = rep_mse_loss(s_cache, t_cache, len_r, rep_layer_indices)

                    # ====== 总损失：按日程组合（解决“刚忘就被修回去”）======
                    loss = (lam_forget * loss_forget
                            + lam_anchor * anchor
                            + lam_rep_forget * loss_rep_forget)

                    if do_repair:
                        loss = loss + (lam_ce * loss_ce
                                       + lam_kd * loss_kd
                                       + lam_rep_retain * loss_rep_retain)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(filter(lambda x: x.requires_grad, model.parameters()), 1.0)
                    optimizer.step()
            # --- 每个 epoch 结束做一次评估 ---
            _, acc_retain, f1_retain = evaluate(model, retain_dataloader, weights, class_num, device)
            _, acc_forget, f1_forget = evaluate(model, forget_dataloader, weights, class_num, device)
            _, acc_test, f1_test = evaluate(model, test_dataloader, weights, class_num, device)

            s = matric_s(before_retain, acc_retain, before_forget, acc_forget)
            mia_now = mia_auc(model, forget_dataloader, test_dataloader, class_num, device,
                              score_type='neg_loss', max_samples=2000, balance=True)

            # -------- 置信度统计（每个epoch）& JS散度（Forget vs Test）--------
            retain_conf = collect_max_conf(model, retain_dataloader, class_num, device, max_samples=2000)
            forget_conf = collect_max_conf(model, forget_dataloader, class_num, device, max_samples=2000)
            test_conf = collect_max_conf(model, test_dataloader, class_num, device, max_samples=2000)
            js_ft_now = js_divergence_from_scores(forget_conf, test_conf, bins=10)
            tab_now = confidence_table(retain_conf, forget_conf, test_conf)

            metrics_history.append({
                "epoch": int(epoch + 1),
                "retain_acc": float(acc_retain),
                "retain_f1": float(f1_retain),
                "forget_acc": float(acc_forget),
                "forget_f1": float(f1_forget),
                "test_acc": float(acc_test),
                "test_f1": float(f1_test),
                "s": float(s),
                "mia_auc": float(mia_now),
                "retain_mean_conf": tab_now["Retain"][0],
                "retain_max_conf": tab_now["Retain"][1],
                "forget_mean_conf": tab_now["Forget"][0],
                "forget_max_conf": tab_now["Forget"][1],
                "test_mean_conf": tab_now["Test"][0],
                "test_max_conf": tab_now["Test"][1],
                "js_forget_test": float(js_ft_now),
            })
            print(f"[Epoch {epoch + 1}/{epochs}] "
                  f"retain_acc={acc_retain:.4f}, retain_f1={f1_retain:.4f}, "
                  f"forget_acc={acc_forget:.4f}, forget_f1={f1_forget:.4f}, "
                  f"test_acc={acc_test:.4f}, test_f1={f1_test:.4f}, s={s:.4f}, mia_auc={mia_now:.4f}")

            # --- 1）按 retain_floor 约束保存 MIA-AUC 最低的模型（越接近0.5越好） ---
            if acc_retain >= retain_floor:
                retain_bad_epochs = 0
                if mia_now < best_mia:
                    best_mia = mia_now
                    bestmia_epoch = epoch + 1
                    torch.save(model.state_dict(), bestmia_savename)
                    print(f"    [Save Best-MIA] epoch={bestmia_epoch}, mia_auc={best_mia:.4f} -> {bestmia_savename}")
            else:
                retain_bad_epochs += 1
                print(
                    f"    [Warn] retain_acc={acc_retain:.4f} < retain_floor={retain_floor:.2f} (bad_epochs={retain_bad_epochs}/{retain_patience})")

            # --- 1.5）若已达到目标遗忘强度（MIA-AUC足够低）且保留满足阈值，则提前停止以节省训练 ---
            if (acc_retain >= retain_floor) and (mia_now <= target_mia):
                print(
                    f"    [EarlyStop] 已达到目标遗忘：mia_auc={mia_now:.4f} <= target_mia={target_mia:.2f} 且 retain_acc={acc_retain:.4f} >= {retain_floor:.2f}")
                break

            # --- 2）自适应调整 k_forget：保留集越稳，就让遗忘步占比越高 ---
            if adaptive_k_forget:
                if acc_retain >= (retain_floor + 0.02) and k_forget < k_forget_max:
                    k_forget += 1
                    print(f"    [Adaptive] retain稳健 -> k_forget 增加到 {k_forget}")
                elif acc_retain < retain_floor and k_forget > 1:
                    k_forget = max(1, k_forget - 1)
                    print(f"    [Adaptive] retain偏低 -> k_forget 降低到 {k_forget}")

            # --- 3）若保留集连续多轮低于阈值，则提前停止，避免精度继续崩 ---
            if retain_bad_epochs >= retain_patience:
                print("    [EarlyStop] 保留精度连续低于阈值，提前结束训练。")
                break

            if s > best_s:
                best_s = s
                torch.save(model.state_dict(), bestsavename)

            result_array = np.vstack((result_array, [acc_retain, acc_forget, acc_test]))

    finally:
        # 移除 forward hooks（避免内存泄漏/重复注册）
        for _h in s_hooks:
            _h.remove()
        for _h in t_hooks:
            _h.remove()

    torch.save(model.state_dict(), lastname)
    return result_array, metrics_history


if __name__ == "__main__":
    epochs = 80
    data = 0
    seed_value = 42
    setup_seed(seed_value)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    num_lays = 6
    class_num = 2
    if class_num == 2:
        weights = torch.FloatTensor([0.50, 0.50])
    elif class_num == 3:
        weights = torch.FloatTensor([0.20, 0.50, 0.30])
    elif class_num == 5:
        weights = torch.FloatTensor([0.05, 0.15, 0.10, 0.50, 0.20])
    else:
        raise ValueError("Unsupported class_num")

    train_path = TRAIN_SPLIT_DIR
    val_path = VAL_DIR
    test_path = TEST_DIR

    val_dataset = PSGDataset(val_path)
    test_dataset = PSGDataset(test_path)
    train_dataset = TrainAllDataset(train_path)

    # 从真实存在的病人目录里取 ID
    patients_np = np.array(
        sorted(
            d for d in os.listdir(train_path)
            if os.path.isdir(os.path.join(train_path, d))
        )
    )
    forget_patients_np = patients_np.copy()

    if len(forget_patients_np) == 0:
        raise RuntimeError(f"No patients found in {train_path}")

    specified_idx = forget_patients_np[data % len(forget_patients_np)]
    print("forget patient:", specified_idx)

    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False, drop_last=False, collate_fn=collate_fn)

    lr_all = 2e-5
    batch_size = 16

    amodel = EBranchformerwithAdapter(
        input_dim=240, output_dim=class_num, encoder_dim=128,
        num_blocks=num_lays, cgmlp_linear_units=256, linear_units=256
    ).to(device)
    amodel.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=device))

    # --- teacher 快照（遗忘前模型，用于 KD + 表示锚定） ---
    teacher_model = copy.deepcopy(amodel).to(device)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    # --- 冻结所有参数，只解冻 adapter（局部更新，减少误伤 retain/test） ---
    for p in amodel.parameters():
        p.requires_grad = False

    # 如果你想更稳：只训练最后 K 层 adapter
    last_k = 2
    for layer_idx in range(num_lays - last_k, num_lays):
        for p in amodel.encoders[layer_idx].adapter.parameters():
            p.requires_grad = True

    # 实验名/日志文件名
    exp_name = f"class2_unlearn_NIR_data{data}_KL{KL_temperature}_lr{lr_all}"
    bestsavename = f"./log1/{exp_name}_best.pth"
    lastname = f"./log1/{exp_name}_last.pth"

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, amodel.parameters()), lr=lr_all)

    forget_path = os.path.join(train_path, specified_idx)
    forget_dataset = PSGDataset(forget_path)
    sampler = get_balanced_sampler(forget_dataset)
    train_forget_dataloader = DataLoader(
        forget_dataset, batch_size=batch_size, sampler=sampler, drop_last=False, collate_fn=collate_fn
    )
    forget_dataloader = DataLoader(forget_dataset, batch_size=256, shuffle=False, drop_last=False,
                                   collate_fn=collate_fn)

    obtain_dataset = TrainDataset(train_path, specified_idx, include=False)
    # retain 评估用 dataloader（大 batch，不 shuffle）
    obtain_dataloader = DataLoader(obtain_dataset, batch_size=256, shuffle=False, drop_last=False,
                                   collate_fn=collate_fn)
    # retain 训练用 dataloader（小 batch，shuffle，用于 repair 分支）
    train_retain_dataloader = DataLoader(obtain_dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                                         collate_fn=collate_fn)

    fig_name = f"2分类data{data}_KL{KL_temperature}_lr{lr_all}"

    # 1）遗忘前：画置信度/分布对比图
    caculate_conf(
        amodel, fig_name, "Before Forgetting",
        forget_dataloader, obtain_dataloader, test_dataloader, class_num, device
    )

    # 2）执行机器遗忘：NIR（Impair–Repair + KD + Anchor + 表示约束）
    unlearn_result, metrics_history = blindspot_unlearner_nir(
        amodel, teacher_model,
        train_forget_dataloader, train_retain_dataloader,
        forget_dataloader, obtain_dataloader, test_dataloader,
        epochs, optimizer, weights,
        class_num, bestsavename, lastname, device
    )

    # 加载 S 指标最佳的权重（你的原评估逻辑）
    # 优先加载“保留精度约束下 MIA-AUC 最低”的权重（更符合机器遗忘目标）
    bestmia_savename = bestsavename.replace('.pth', '_bestmia.pth')
    if os.path.exists(bestmia_savename):
        print(f"[Load] 使用 Best-MIA 权重: {bestmia_savename}")
        amodel.load_state_dict(torch.load(bestmia_savename, map_location=device))
    else:
        print(f"[Load] 未找到 Best-MIA 权重，退回加载 best_s: {bestsavename}")
        amodel.load_state_dict(torch.load(bestsavename, map_location=device))

    # -------- Final-BestMIA：置信度表格 & JS散度（Forget vs Test）& MIA-AUC --------
    mia_final = mia_auc(amodel, forget_dataloader, test_dataloader, class_num, device,
                        score_type='neg_loss', max_samples=2000, balance=True)
    retain_conf_f = collect_max_conf(amodel, obtain_dataloader, class_num, device, max_samples=2000)
    forget_conf_f = collect_max_conf(amodel, forget_dataloader, class_num, device, max_samples=2000)
    test_conf_f = collect_max_conf(amodel, test_dataloader, class_num, device, max_samples=2000)
    js_ft_final = js_divergence_from_scores(forget_conf_f, test_conf_f, bins=10)
    tab_f = confidence_table(retain_conf_f, forget_conf_f, test_conf_f)
    print_conf_table("Final-BestMIA-Metrics", tab_f, js_ft_final)
    print(f"[Final-BestMIA] mia_auc={mia_final:.4f}  （越接近0.5越好）")
    # -------- 保存 metrics：CSV + 曲线图PNG（与 bestmia 权重同目录）--------
    # 额外再评估一次 bestmia 模型的 acc/f1，便于论文直接对比
    _, final_retain_acc, final_retain_f1 = evaluate(amodel, obtain_dataloader, weights, class_num, device)
    _, final_forget_acc, final_forget_f1 = evaluate(amodel, forget_dataloader, weights, class_num, device)
    _, final_test_acc, final_test_f1 = evaluate(amodel, test_dataloader, weights, class_num, device)

    # 将 Final-BestMIA 的指标写入一行，形成“Before vs After”的可复现实验记录
    metrics_history.append({
        "epoch": "final_bestmia",
        "retain_acc": float(final_retain_acc),
        "retain_f1": float(final_retain_f1),
        "forget_acc": float(final_forget_acc),
        "forget_f1": float(final_forget_f1),
        "test_acc": float(final_test_acc),
        "test_f1": float(final_test_f1),
        "s": "",
        "mia_auc": float(mia_final),
        "retain_mean_conf": tab_f["Retain"][0],
        "retain_max_conf": tab_f["Retain"][1],
        "forget_mean_conf": tab_f["Forget"][0],
        "forget_max_conf": tab_f["Forget"][1],
        "test_mean_conf": tab_f["Test"][0],
        "test_max_conf": tab_f["Test"][1],
        "js_forget_test": float(js_ft_final),
    })

    # metrics 文件建议与实际采用的权重同名前缀（bestmia 优先），避免和 last/best 混淆
    metrics_prefix = bestmia_savename if os.path.exists(bestmia_savename) else bestsavename
    metrics_csv = metrics_prefix.replace('.pth', '_metrics.csv')
    metrics_png = metrics_prefix.replace('.pth', '_metrics.png')
    save_metrics_history_csv(metrics_csv, metrics_history)
    plot_metrics_curves(metrics_png, metrics_history)
    print(f"[Saved] metrics_csv -> {metrics_csv}")
    print(f"[Saved] metrics_png -> {metrics_png}")
    # 3）遗忘后：再画一张图做 before/after 对比
    caculate_conf(
        amodel, fig_name, "After Forgetting",
        forget_dataloader, obtain_dataloader, test_dataloader, class_num, device
    )
