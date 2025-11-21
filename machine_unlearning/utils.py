import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
import seaborn as sns

KL_temperature = 1


def get_balanced_sampler(dataset):
    labels = [item[1] for item in dataset]
    label_counts = Counter(labels)
    weights = {label: 1.0 / count for label, count in label_counts.items()}
    sample_weights = [weights[label] for label in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(dataset), replacement=True)
    return sampler


def updata_model(model, parameters_to_update):
    i = 0
    for layer_idx in range(6):
        adapter_params = model.encoders[layer_idx].adapter.parameters()
        for param in adapter_params:
            if i < len(parameters_to_update):
                param.data.copy_(parameters_to_update[i].data)
                i = i + 1
            else:
                break


def init_weights(amodel):
    for layer_idx in range(6):
        adapter_params = amodel.encoders[layer_idx].adapter.parameters()
        for param in adapter_params:
            if param.requires_grad:
                if param.dim() >= 2:
                    nn.init.xavier_uniform_(param)


def convert_labels(labels, class_num):
    if class_num == 2:
        class_dict = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1}
    elif class_num == 3:
        class_dict = {0: 0, 1: 1, 2: 2, 3: 2, 4: 2}
    elif class_num == 5:
        return labels  # 5 分类时，直接返回原始标签
    else:
        raise ValueError("class_num 只能是 2, 3, 或 5")
    return [class_dict[label] for label in labels]


def matric_s(before_retain_acc, after_retain_acc, before_forget_acc, after_forget_acc):
    s_r = np.exp(after_retain_acc - before_retain_acc)
    s_f = np.exp(after_forget_acc - before_forget_acc)
    a = 1
    b = -1
    s = s_r ** a * s_f ** b
    return s


def plot_training(loss_list):
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(loss_list[:, 0], color='blue', label="Train_loss")
    ax[0].plot(loss_list[:, 1], color='green', label="Validation_Loss")

    ax[0].set_title("Loss Curves - " + 'loss', fontsize=12)
    ax[0].set_ylabel("Loss", fontsize=10)
    ax[0].set_xlabel("Epoch", fontsize=10)
    ax[1].plot(loss_list[:, 2], color='red', label="Validation_accuracy")
    ax[1].set_title("ACC Curves", fontsize=12)
    ax[1].set_ylabel("acc", fontsize=10)
    ax[1].set_xlabel("Epoch", fontsize=10)

    plt.tight_layout()

    plt.show()


def plot_accuracy(unlearn_result):
    plt.plot(unlearn_result[:, 0], label=f'Retained Data')
    plt.plot(unlearn_result[:, 1], label=f'Forgotten Data')
    plt.plot(unlearn_result[:, 2], label=f'Test Data')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Variation of Each Dimension')

    plt.xticks(np.arange(0, len(unlearn_result), step=1))
    plt.show()


def percentage(data, percent_ranges):
    total_count = len(data)
    percent_counts = []
    all_per = []
    for start, end in percent_ranges:
        count = np.sum(np.logical_and(np.array(data * 100) >= start, np.array(data * 100) < end))
        percent = (count / total_count) * 100
        percent_counts.append((start, end, percent))
        all_per.append(percent)
    return all_per


def collect_prob(dataloader, model, class_num, device):
    prob = np.empty((0, class_num))
    confidence_max = np.array([])
    model.eval()
    with torch.no_grad():
        for batch_id, (wavs, labels, _, _) in enumerate(dataloader):
            wavs = torch.tensor(np.array(wavs)).to(device)
            features_len = torch.IntTensor([len(feat) for feat in wavs]).to(device)
            features = pad_sequence(wavs, batch_first=True)
            predicted = model(features, features_len)
            predicted_softmax = F.softmax(predicted, dim=1)
            prob = np.vstack((prob, predicted_softmax.to("cpu")))
            predicted_conf, predicted_classid = torch.max(predicted_softmax, dim=1)  # TODO: improve the confidence
            confidence_max = np.concatenate((confidence_max, predicted_conf.detach().cpu().numpy()))

    return prob, confidence_max


def plot_bar(data1, data2, data3, name, state):
    categories = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100']
    bar_width = 0.20

    bar_positions1 = np.arange(len(categories))
    bar_positions2 = bar_positions1 + bar_width
    bar_positions3 = bar_positions2 + bar_width

    plt.bar(bar_positions1, data1, width=bar_width, label='Retain Set', color="#74AED4")
    plt.bar(bar_positions2, data2, width=bar_width, label='Forget Set', color="#F1766D")
    plt.bar(bar_positions3, data3, width=bar_width, label='Test Set', color='#CEAFD4')

    plt.ylim(0, 80)
    plt.xlabel('Confidence Intervals')
    plt.ylabel('Ratio Distribution')
    plt.xticks(bar_positions1 + bar_width / 2, categories)  # 设置x轴刻度位置
    plt.title(f"Max Confidence Score Distribution {state}", fontsize=14)
    # 显示图例
    plt.legend()
    plt.savefig(f"{name} {state}柱状图.png", format="png", dpi=600, bbox_inches="tight")
    plt.close()


def caculate_conf(model, name, state, forget_dataloader, retain_dataloader, test_dataloader, class_num, device):
    retain_prob, retain_conf = collect_prob(retain_dataloader, model, class_num, device)
    forget_prob, forget_conf = collect_prob(forget_dataloader, model, class_num, device)
    test_prob, test_conf = collect_prob(test_dataloader, model, class_num, device)
    percent_ranges = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90),
                      (90, 100)]

    after_precent_retain = percentage(retain_conf, percent_ranges)
    after_precent_forget = percentage(forget_conf, percent_ranges)
    after_precent_test = percentage(test_conf, percent_ranges)
    plot_bar(after_precent_retain, after_precent_forget, after_precent_test, name, state)


def plot_confidence(name, retain_confidences, for_confidences, test_confidences):
    # 创建直方图
    plt.figure(figsize=(10, 6))
    sns.histplot(retain_confidences, label='Retain Set', kde=True, color='blue', bins=20, stat='percent', alpha=0.6)
    sns.histplot(for_confidences, label='Forget Set', kde=True, color='green', bins=20, stat='percent', alpha=0.6)
    sns.histplot(test_confidences, label='Test Set', kde=True, color='orange', bins=20, stat='percent', alpha=0.6)
    plt.title(f"Max Confidence Score Distribution{name}", fontsize=16, weight='bold')
    plt.xlabel('Confidence Score', fontsize=14)
    plt.ylabel('Percentage (%)', fontsize=14)  # 将 y 轴标签改为 percent
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.legend(title='Dataset', fontsize=12, title_fontsize=13)

    plt.tight_layout()

    plt.show()
