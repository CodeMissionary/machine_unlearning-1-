import time
import torch
import torch.nn as nn
import sys
from ebranchformer import EBranchformer, EBranchformerwithAdapter
import random
import torchaudio
from torch.nn import functional as F
import copy
from tqdm import tqdm
import math
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from s3prl.utility.helper import hack_isinstance
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

torch.multiprocessing.set_sharing_strategy('file_system')
torchaudio.set_audio_backend('sox_io')
hack_isinstance()
import torch.optim.lr_scheduler as lr_scheduler
from collections import defaultdict

from data import *
from data.process_data.dataset import collate_fn
from data.process_data.extract_fbank_feature import *
from utils import *

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # /home/mzk/machine_unlearning
DATA_ROOT = os.path.join(BASE_DIR, "data", "all_data")
TRAIN_SPLIT_DIR = os.path.join(DATA_ROOT, "train_split_by_patient")
VAL_DIR = os.path.join(DATA_ROOT, "validation")
TEST_DIR = os.path.join(DATA_ROOT, "test")


def train_one_epoch(model, dataloader, optimizer, weights, class_num, device):
    model.train()
    losses = []
    backward_steps = 0
    weights = weights.to(device)
    for batch_id, (wavs, labels, filenames, idx) in enumerate(dataloader):
        wavs = torch.tensor(np.array(wavs)).to(device)
        features_len = torch.IntTensor([len(feat) for feat in wavs]).to(device)
        features = pad_sequence(wavs, batch_first=True)
        predicted = model(features, features_len)
        predicted_softmax = F.softmax(predicted, dim=1)
        labels = convert_labels(labels, class_num)
        labels = torch.LongTensor(labels).to(device)
        CE_loss = F.cross_entropy(predicted_softmax, labels, reduction='none')
        pt = torch.exp(-CE_loss)
        gamma = 1
        FL_loss = weights[labels] * (1 - pt) ** gamma * CE_loss
        loss = torch.mean(FL_loss)
        loss.backward()

        grads = [p.grad for p in filter(lambda p: p.requires_grad, model.parameters()) if p.grad is not None]
        grad_norm = torch.nn.utils.clip_grad_norm_(grads, 1)
        if math.isnan(grad_norm):
            print(f'[Runner] - grad norm is NaN at step {global_step}')
        else:
            optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    return np.mean(losses)


def meta_adapter_train(amodel, random_model, sup_forget_dataloader, sup_obtain_dataloader,
                       query_forget_dataloader, query_obtain_dataloader, num_lays, weights, device):
    copymodel = copy.deepcopy(amodel)
    copymodel.train()

    for param in copymodel.parameters():
        param.requires_grad = False

    for layer_idx in range(num_lays):
        adapter_params = copymodel.encoders[layer_idx].adapter.parameters()
        for param in adapter_params:
            param.requires_grad = True
    weights = weights.to(device)
    fast_weights = [param for name, param in copymodel.named_parameters() if param.requires_grad]
    inner_optimizer = torch.optim.Adam(fast_weights, lr=1e-4)
    for step, (wavs, labels, _, _) in enumerate(sup_forget_dataloader):
        wavs = torch.tensor(np.array(wavs)).to(device)
        features_len = torch.IntTensor([len(feat) for feat in wavs]).to(device)
        features = pad_sequence(wavs, batch_first=True)
        model_output_pred = copymodel(features, features_len)
        with torch.no_grad():
            ranmodel_output_pred = random_model(features, features_len)
        KL_temperature = 1
        loss = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(model_output_pred / KL_temperature, dim=1),
                                          torch.nn.functional.softmax(ranmodel_output_pred / KL_temperature, dim=1),
                                          reduction='batchmean')
        inner_optimizer.zero_grad()
        loss.backward()
        inner_optimizer.step()

    for step, (wavs, labels, _, _) in enumerate(sup_obtain_dataloader):
        wavs = torch.tensor(np.array(wavs)).to(device)
        features_len = torch.IntTensor([len(feat) for feat in wavs]).to(device)
        features = pad_sequence(wavs, batch_first=True)
        model_output_pred = copymodel(features, features_len)
        predicted_softmax = F.softmax(model_output_pred, dim=1)
        labels = convert_labels(labels, class_num)
        labels = torch.LongTensor(labels).to(device)
        CE_loss = F.cross_entropy(predicted_softmax, labels, reduction='none')
        pt = torch.exp(-CE_loss)
        gamma = 1
        FL_loss = weights[labels] * (1 - pt) ** gamma * CE_loss
        loss = torch.mean(FL_loss)
        inner_optimizer.zero_grad()
        loss.backward()
        inner_optimizer.step()

    for step, (wavs, labels, _, _) in enumerate(query_forget_dataloader):
        wavs = torch.tensor(np.array(wavs)).to(device)
        features_len = torch.IntTensor([len(feat) for feat in wavs]).to(device)
        features = pad_sequence(wavs, batch_first=True)
        model_output_pred = copymodel(features, features_len)
        with torch.no_grad():
            ranmodel_output_pred = random_model(features, features_len)
        KL_temperature = 1
        loss = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(model_output_pred / KL_temperature, dim=1),
                                          torch.nn.functional.softmax(ranmodel_output_pred / KL_temperature, dim=1),
                                          reduction='batchmean')
        inner_optimizer.zero_grad()
        loss.backward()
        inner_optimizer.step()

    accumulated_grads = {p: torch.zeros_like(p) for p in fast_weights}
    for step, (wavs, labels, _, _) in enumerate(query_forget_dataloader):
        wavs = torch.tensor(np.array(wavs)).to(device)
        features_len = torch.IntTensor([len(feat) for feat in wavs]).to(device)
        features = pad_sequence(wavs, batch_first=True)
        model_output_pred = copymodel(features, features_len)
        with torch.no_grad():
            ranmodel_output_pred = random_model(features, features_len)
        KL_temperature = 1
        loss = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(model_output_pred / KL_temperature, dim=1),
                                          torch.nn.functional.softmax(ranmodel_output_pred / KL_temperature, dim=1),
                                          reduction='batchmean')
        grads = torch.autograd.grad(loss, fast_weights)
        for p, g in zip(fast_weights, grads):
            accumulated_grads[p] += g

    for step, (wavs, labels, _, _) in enumerate(query_obtain_dataloader):
        wavs = torch.tensor(np.array(wavs)).to(device)
        features_len = torch.IntTensor([len(feat) for feat in wavs]).to(device)
        features = pad_sequence(wavs, batch_first=True)
        predicted = copymodel(features, features_len)
        predicted_softmax = F.softmax(predicted, dim=1)
        labels = convert_labels(labels, class_num)
        labels = torch.LongTensor(labels).to(device)
        CE_loss = F.cross_entropy(predicted_softmax, labels, reduction='none')
        pt = torch.exp(-CE_loss)
        gamma = 1
        FL_loss = weights[labels] * (1 - pt) ** gamma * CE_loss
        loss = torch.mean(FL_loss)
        grads = torch.autograd.grad(loss, fast_weights)
        for p, g in zip(fast_weights, grads):
            accumulated_grads[p] += g

    return accumulated_grads


def evaluate(model, dataloader, weights, class_num, device):
    model.eval()
    losses = []
    true_labels = []
    predicted_labels = []
    weights = weights.to(device)
    with torch.no_grad():
        for batch_id, (wavs, labels, _, _) in enumerate(dataloader):
            wavs = torch.tensor(np.array(wavs)).to(device)
            features_len = torch.IntTensor([len(feat) for feat in wavs]).to(device)
            features = pad_sequence(wavs, batch_first=True)
            predicted = model(features, features_len)
            predicted_softmax = F.softmax(predicted, dim=1)
            predicted_conf, predicted_classid = torch.max(predicted_softmax, dim=1)  # TODO: improve the confidence
            labels = convert_labels(labels, class_num)
            labels = torch.LongTensor(labels).to(device)
            CE_loss = F.cross_entropy(predicted_softmax, labels, reduction='none')
            pt = torch.exp(-CE_loss)
            gamma = 1
            FL_loss = weights[labels] * (1 - pt) ** gamma * CE_loss
            loss = torch.mean(FL_loss)
            losses.append(loss.item())
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted_classid.cpu().numpy())
    accuracy = accuracy_score(np.array(true_labels), np.array(predicted_labels))
    macro_f1 = f1_score(np.array(true_labels), np.array(predicted_labels), average='macro')

    return np.mean(losses), accuracy, macro_f1


def pre_train(model, train_dataloader, val_dataloader, epochs, best_premodel, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)
    loss_list = []
    best_acc = 0
    for epoch in tqdm(range(epochs)):
        train_loss = train_one_epoch(model, train_dataloader, optimizer, weights, class_num, device)
        val_loss, acc, f1 = evaluate(model, val_dataloader, weights, class_num, device)
        sched.step(val_loss)
        loss_list.append([train_loss, val_loss, acc, f1])
        if acc > best_acc:
            torch.save(model.state_dict(), best_premodel)
            best_acc = acc
    loss_array = np.array(loss_list)
    print(f"pretrain loss_array： {loss_array}")


def adapter_train_one_epoch(amodel, random_model, num_lays, weights, patients_np, update_lr, device):
    # 用我们在文件顶部定义的 TRAIN_SPLIT_DIR
    train_path = TRAIN_SPLIT_DIR

    # 实际有多少个病人（= train_split_by_patient 下面有多少个子目录）
    num_patients = len(patients_np)
    if num_patients == 0:
        raise RuntimeError(f"No patients found in {train_path}")

    # 元训练迭代次数，最多 50 次，病人少就用病人数
    num_meta_iters = min(50, num_patients)

    fast_weights = [param for name, param in amodel.named_parameters() if param.requires_grad]

    # ===== 元训练循环 =====
    for i in range(num_meta_iters):
        # ---------- support（忘记集） ----------
        sup_specified_idx = patients_np[i % num_patients]
        sup_forget_path = os.path.join(train_path, sup_specified_idx)
        sup_forget_dataset = PSGDataset(sup_forget_path)
        sampler = get_balanced_sampler(sup_forget_dataset)
        sup_forget_dataloader = DataLoader(
            sup_forget_dataset, batch_size=64, sampler=sampler,
            drop_last=False, collate_fn=collate_fn
        )

        # ---------- support（保留集） ----------
        sup_obtain_idx = patients_np[np.random.randint(0, num_patients)]
        sup_obtain_path = os.path.join(train_path, sup_obtain_idx)
        sup_obtain_dataset = PSGDataset(sup_obtain_path)
        sup_obtain_dataloader = DataLoader(
            sup_obtain_dataset, batch_size=64, shuffle=True,
            drop_last=False, collate_fn=collate_fn
        )

        # ---------- query（忘记集） ----------
        rid = np.random.randint(0, num_patients)
        query_specified_idx = patients_np[rid]
        query_forget_path = os.path.join(train_path, query_specified_idx)
        query_forget_dataset = PSGDataset(query_forget_path)
        sampler = get_balanced_sampler(query_forget_dataset)
        query_forget_dataloader = DataLoader(
            query_forget_dataset, batch_size=64, sampler=sampler,
            drop_last=False, collate_fn=collate_fn
        )

        # ---------- query（保留集） ----------
        query_obtain_idx = patients_np[np.random.randint(0, num_patients)]
        query_obtain_path = os.path.join(train_path, query_obtain_idx)
        query_obtain_dataset = PSGDataset(query_obtain_path)
        query_obtain_dataloader = DataLoader(
            query_obtain_dataset, batch_size=64, shuffle=True,
            drop_last=False, collate_fn=collate_fn
        )

        grads = meta_adapter_train(amodel, random_model, sup_forget_dataloader, sup_obtain_dataloader,
                                   query_forget_dataloader, query_obtain_dataloader, num_lays, weights, device)
        fast_weights = list(map(lambda p: p[1] - update_lr * p[0], zip(grads, fast_weights)))
        updata_model(amodel, fast_weights)
        fast_weights = [param for name, param in amodel.named_parameters() if param.requires_grad]

    # 元测试
    meta_matric = 0
    num_patients = len(patients_np)
    if num_patients == 0:
        raise RuntimeError(f"No patients found in {train_path}")

    # 最多评估 12 次，但不能超过病人数量
    num_eval_tasks = min(12, num_patients)

    for i in range(num_eval_tasks):
        # 选一个要“忘记”的病人：按顺序或者循环取
        specified_idx = patients_np[i % num_patients]

        patient_path = os.path.join(train_path, specified_idx)
        patient_dataset = PSGDataset(patient_path)

        sampler = get_balanced_sampler(patient_dataset)
        train_patient_dataloader = DataLoader(
            patient_dataset, batch_size=64, sampler=sampler,
            drop_last=False, collate_fn=collate_fn
        )

        patient_dataloader = DataLoader(
            patient_dataset, batch_size=128, shuffle=True,
            drop_last=False, collate_fn=collate_fn
        )

        # 再随机选一个“保留”的病人（也只能在 0 ~ num_patients-1 里选）
        obtain_idx = patients_np[random.randint(0, num_patients - 1)]
        obtain_path = os.path.join(train_path, obtain_idx)
        obtain_dataset = PSGDataset(obtain_path)
        obtain_dataloader = DataLoader(
            obtain_dataset, batch_size=128, shuffle=True,
            drop_last=False, collate_fn=collate_fn
        )

        copymodel = amodel
        copymodel.load_state_dict(amodel.state_dict())
        for param in copymodel.parameters():
            param.requires_grad = False

        for layer_idx in range(num_lays):
            adapter_params = copymodel.encoders[layer_idx].adapter.parameters()
            for param in adapter_params:
                param.requires_grad = True
        _, beforeforgetacc, _ = evaluate(copymodel, patient_dataloader, weights, class_num, device)
        _, beforeobtainacc, _ = evaluate(copymodel, obtain_dataloader, weights, class_num, device)
        inner_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, copymodel.parameters()), lr=1e-4)
        for step, (wavs, labels, _, _) in enumerate(train_patient_dataloader):
            wavs = torch.tensor(np.array(wavs)).to(device)
            features_len = torch.IntTensor([len(feat) for feat in wavs]).to(device)
            features = pad_sequence(wavs, batch_first=True)
            model_output_pred = copymodel(features, features_len)
            with torch.no_grad():
                ranmodel_output_pred = random_model(features, features_len)
            KL_temperature = 1
            loss = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(model_output_pred / KL_temperature, dim=1),
                torch.nn.functional.softmax(ranmodel_output_pred / KL_temperature, dim=1),
                reduction='batchmean')
            inner_optimizer.zero_grad()
            loss.backward()
            inner_optimizer.step()

        _, afterforgetacc, _ = evaluate(copymodel, patient_dataloader, weights, class_num, device)
        _, afterobtainacc, _ = evaluate(copymodel, obtain_dataloader, weights, class_num, device)
        meta_matric += matric_s(beforeobtainacc, afterobtainacc, beforeforgetacc, afterforgetacc)
    return meta_matric / 12


def meta_train(loaded_model, amodel, random_model, train_dataloader, val_dataloader, epochs,
               num_lays, weights,
               lr_adapter, bestname, savename, device):
    # 1 先训练不带adapter的模型
    best_premodel = './log1/class2_pretrain_epoch15_b32_best.pth'
    pre_train(loaded_model, train_dataloader, val_dataloader, 15, best_premodel, device)
    # loaded_model.load_state_dict(torch.load(best_premodel, map_location=device))

    # 2 插入adapter
    model_state_dict = loaded_model.state_dict()
    amodel_state_dict = amodel.state_dict()
    for name, param in amodel_state_dict.items():
        corresponding_param = model_state_dict.get(name, None)
        if corresponding_param is not None:
            amodel_state_dict[name] = corresponding_param
    amodel.load_state_dict(amodel_state_dict)

    # 3 冻结所有参数#
    for param in amodel.parameters():
        param.requires_grad = True
    # 选择性地解冻一些参数
    for layer_idx in range(num_lays):
        adapter_params = amodel.encoders[layer_idx].adapter.parameters()
        for param in adapter_params:
            param.requires_grad = False

    writer = SummaryWriter()
    losses = []
    patients_np = np.array(
        sorted(
            d for d in os.listdir(TRAIN_SPLIT_DIR)
            if os.path.isdir(os.path.join(TRAIN_SPLIT_DIR, d))
        )
    )
    print("patients_np:", patients_np)
    train_s_matric = 0
    weights = weights.to(device)
    # 4 交替训练
    for epoch in tqdm(range(epochs)):
        # 4.1  训练adapter
        for param in amodel.parameters():
            param.requires_grad = False
        for layer_idx in range(num_lays):
            adapter_params = amodel.encoders[layer_idx].adapter.parameters()
            for param in adapter_params:
                param.requires_grad = True
        s_matric = adapter_train_one_epoch(amodel, random_model, num_lays, weights, patients_np,
                                           lr_adapter,
                                           device)
        print(f"s_matric {s_matric}")
        # ----------------------------------------------------------------------------------------------------
        # 4.2 训练主体

        for param in amodel.parameters():
            param.requires_grad = True
        for layer_idx in range(num_lays):
            adapter_params = amodel.encoders[layer_idx].adapter.parameters()
            for param in adapter_params:
                param.requires_grad = False
        optimizer_all = torch.optim.Adam(filter(lambda p: p.requires_grad, amodel.parameters()), lr=5e-6)
        amodel.train()
        for step, (wavs, labels, _, _) in enumerate(train_dataloader):
            wavs = torch.tensor(np.array(wavs)).to(device)
            features_len = torch.IntTensor([len(feat) for feat in wavs]).to(device)
            features = pad_sequence(wavs, batch_first=True)
            predicted = amodel(features, features_len)
            predicted_softmax = F.softmax(predicted, dim=1)
            labels = convert_labels(labels, class_num)
            labels = torch.LongTensor(labels).to(device)
            CE_loss = F.cross_entropy(predicted_softmax, labels, reduction='none')
            pt = torch.exp(-CE_loss)
            gamma = 1
            FL_loss = weights[labels] * (1 - pt) ** gamma * CE_loss
            optimizer_all.zero_grad()
            loss = torch.mean(FL_loss)
            loss.backward()
            optimizer_all.step()

        val_loss, acc, f1 = evaluate(amodel, val_dataloader, weights, class_num, device)
        losses.append([s_matric, val_loss, acc])

        for name, param in amodel.named_parameters():
            writer.add_histogram(name, param.clone().detach().cpu().numpy(), epoch + 1)
        writer.add_scalars('s_matric', {'s': s_matric}, epoch + 1)
        writer.add_scalars('valloss', {'loss': val_loss}, epoch + 1)
        writer.add_scalars('valacc', {'acc': acc}, epoch + 1)
        if (s_matric > train_s_matric):
            train_s_matric = s_matric
            torch.save(amodel.state_dict(), bestname)
    writer.close()
    np_losses = np.array(losses)
    plot_training(np_losses)
    torch.save(amodel.state_dict(), savename)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    lr_adapter = 1e-5
    epochs = 10
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
    model = EBranchformer(input_dim=240, output_dim=class_num, encoder_dim=128, num_blocks=num_lays,
                          cgmlp_linear_units=256, linear_units=256).to(device)
    amodel = EBranchformerwithAdapter(input_dim=240, output_dim=class_num, encoder_dim=128, num_blocks=num_lays,
                                      cgmlp_linear_units=256, linear_units=256).to(device)
    random_model = EBranchformerwithAdapter(input_dim=240, output_dim=class_num, encoder_dim=128, num_blocks=num_lays,
                                            cgmlp_linear_units=256, linear_units=256).to(device)
    random_model.eval()

    val_path = VAL_DIR
    test_path = TEST_DIR
    train_path = TRAIN_SPLIT_DIR
    val_dataset = PSGDataset(val_path)
    test_dataset = PSGDataset(test_path)
    train_dataset = TrainAllDataset(train_path)

    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=False, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False, drop_last=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False, drop_last=False, collate_fn=collate_fn)

    savename = "./log1/class2_pre_e10_b128_sloss1_1_adapter3.pth"
    bestname = "./log1/class2_pre_e10_b128_sloss1_1_adapter3_best.pth"
    meta_train(model, amodel, random_model, train_dataloader, val_dataloader, epochs, num_lays,
               weights, lr_adapter,
               bestname, savename, device)
