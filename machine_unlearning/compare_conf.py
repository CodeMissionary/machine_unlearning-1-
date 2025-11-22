import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import numpy
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR, ExponentialLR
import sys
import matplotlib.pyplot as plt
from train import *
from data.process_data.extract_fbank_feature import *
import seaborn as sns
# from metric.memershipattack import *
from ebranchformer import *


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_confidence(name, train_confidences, val_confidences, test_confidences):
    plt.figure(figsize=(10, 6))
    sns.histplot(train_confidences, label='Train Set', kde=True, color='blue', bins=20, stat='percent')
    sns.histplot(val_confidences, label='Val Set', kde=True, color='green', bins=20, stat='percent')
    sns.histplot(test_confidences, label='Test Set', kde=True, color='orange', bins=20, stat='percent')
    plt.title(name)
    plt.xlabel('Confidence Score')
    plt.ylabel('percent')
    plt.legend()
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


def plot_bar(data1, data2, data3, name):
    categories = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100']
    bar_width = 0.20
    bar_positions1 = np.arange(len(categories))
    bar_positions2 = bar_positions1 + bar_width
    bar_positions3 = bar_positions2 + bar_width
    plt.bar(bar_positions1, data1, width=bar_width, label='Train Data')
    plt.bar(bar_positions2, data2, width=bar_width, label='Val Data')
    plt.bar(bar_positions3, data3, width=bar_width, label='Test Data')
    plt.xlabel('Confidence Intervals')
    plt.ylabel('Ratio Distribution')
    plt.xticks(bar_positions1 + bar_width / 2, categories)  # 设置x轴刻度位置
    plt.title(name)
    plt.legend()
    plt.show()


def caculate_conf(model, retain_dataloader, forget_dataloader, test_dataloader, class_num, device):
    retain_prob, retain_conf = collect_prob(retain_dataloader, model, class_num, device)
    forget_prob, forget_conf = collect_prob(forget_dataloader, model, class_num, device)
    test_prob, test_conf = collect_prob(test_dataloader, model, class_num, device)

    plot_confidence('max Confidence Score Distribution', retain_conf, forget_conf, test_conf)
    percent_ranges = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90),
                      (90, 100)]
    after_precent_train = percentage(train_conf, percent_ranges)
    after_precent_val = percentage(val_conf, percent_ranges)
    after_precent_test = percentage(test_conf, percent_ranges)
    plot_bar(after_precent_train, after_precent_val, after_precent_test, "after_precent_")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
