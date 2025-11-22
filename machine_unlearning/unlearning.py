import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import numpy
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR, ExponentialLR
import sys
import seaborn as sns
import random
from utils import *
from ebranchformer import *
from data.process_data.extract_fbank_feature import *
# from metric.memershipattack import *
from torch.nn.utils.rnn import pad_sequence
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "data", "all_data")
TRAIN_SPLIT_DIR = os.path.join(DATA_ROOT, "train_split_by_patient")
VAL_DIR = os.path.join(DATA_ROOT, "validation")
TEST_DIR = os.path.join(DATA_ROOT, "test")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, dataloader, weights, class_num, device):
    model.eval()
    losses = []
    true_labels = []
    predicted_labels = []
    weights = weights.to(device)
    with torch.no_grad():
        for batch_id, (wavs, labels, _, _) in enumerate(dataloader):
            wavs = torch.tensor(np.array(wavs)).to(device)
            labels = convert_labels(labels, class_num)
            labels = torch.LongTensor(labels).to(device)

            features_len = torch.IntTensor([len(feat) for feat in wavs]).to(device)
            features = pad_sequence(wavs, batch_first=True)
            predicted = model(features, features_len)
            predicted_softmax = F.softmax(predicted, dim=1)
            predicted_conf, predicted_classid = torch.max(predicted_softmax, dim=1)  # TODO: improve the confidence
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


def blindspot_unlearner(model, random_model, train_forget_dataloader, forget_dataloader,
                        retain_dataloader,
                        test_dataloader, epochs, optimizer, weights, class_num, bestsavename, lastname, device):
    result_array = numpy.empty((0, 3))
    _, accretain, _ = evaluate(model, retain_dataloader, weights, class_num, device)
    _, accfor, _ = evaluate(model, forget_dataloader, weights, class_num, device)
    _, acctest, _ = evaluate(model, test_dataloader, weights, class_num, device)
    result_array = np.vstack((result_array, [accretain, accfor, acctest]))
    beforeobtainacc = accretain
    beforeforgetacc = accfor
    best_s = 0
    for epoch in range(epochs):
        model.train()
        for batch_id, (wavs, labels, _, _) in enumerate(train_forget_dataloader):
            wavs = torch.tensor(np.array(wavs)).to(device)
            features_len = torch.IntTensor([len(feat) for feat in wavs]).to(device)
            features = pad_sequence(wavs, batch_first=True)
            model_output_pred = model(features, features_len)
            ranmodel_output_pred = random_model(features, features_len)
            loss = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(model_output_pred / KL_temperature, dim=1),
                torch.nn.functional.softmax(ranmodel_output_pred / KL_temperature, dim=1), reduction='batchmean')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        _, accretain, _ = evaluate(model, retain_dataloader, weights, class_num, device)
        _, accfor, _ = evaluate(model, forget_dataloader, weights, class_num, device)
        _, acctest, _ = evaluate(model, test_dataloader, weights, class_num, device)

        s = matric_s(beforeobtainacc, accretain, beforeforgetacc, accfor)

        if s > best_s:
            best_s = s
            torch.save(model.state_dict(), bestsavename)
        result_array = np.vstack((result_array, [accretain, accfor, acctest]))

    torch.save(model.state_dict(), lastname)
    return result_array


if __name__ == "__main__":
    epochs = 10
    data = 9
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
    train_path = TRAIN_SPLIT_DIR
    val_path = VAL_DIR
    test_path = TEST_DIR

    val_dataset = PSGDataset(val_path)
    test_dataset = PSGDataset(test_path)
    train_dataset = TrainAllDataset(train_path)

    patients_np = np.load('./data/process_data/patient_ids.npy')
    forget_patients_np = np.load('./data/process_data/forget_patients.npy')

    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False, drop_last=False, collate_fn=collate_fn)
    lr_all = 4e-5
    batch_size = 16
    amodel = EBranchformerwithAdapter(input_dim=240, output_dim=class_num, encoder_dim=128, num_blocks=num_lays,
                                      cgmlp_linear_units=256, linear_units=256).to(device)
    random_model = EBranchformerwithAdapter(input_dim=240, output_dim=class_num, encoder_dim=128,
                                            num_blocks=num_lays,
                                            cgmlp_linear_units=256, linear_units=256).to(device)
    random_model.eval()
    amodel.load_state_dict(torch.load('./log1/class2_pre_e10_b128_sloss1_1_adapter1_best.pth', map_location=device))

    for param in amodel.parameters():
        param.requires_grad = False

    for layer_idx in range(num_lays):
        adapter_params = amodel.encoders[layer_idx].adapter.parameters()
        for param in adapter_params:
            param.requires_grad = True

    bestsavename = './log1/forget/class2_pre_e10_b128_sloss1_1_adapter1_best_data{}_batch{}_lr{}.pth'.format(data,
                                                                                                             batch_size,
                                                                                                             lr_all)
    lastname = './log1/forget/class2_pre_e10_b128_sloss1_1_adapter1_last_data{}_batch{}_lr{}.pth'.format(data,
                                                                                                         batch_size,
                                                                                                         lr_all)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, amodel.parameters()), lr=lr_all)

    specified_idx = forget_patients_np[data]
    print(specified_idx)
    forget_path = os.path.join(train_path, specified_idx)
    forget_dataset = PSGDataset(forget_path)
    sampler = get_balanced_sampler(forget_dataset)
    train_forget_dataloader = DataLoader(forget_dataset, batch_size=batch_size, sampler=sampler, drop_last=False,
                                         collate_fn=collate_fn)

    forget_dataloader = DataLoader(forget_dataset, batch_size=256, shuffle=False, drop_last=False,
                                   collate_fn=collate_fn)
    obtain_dataset = TrainDataset(train_path, specified_idx, include=False)
    obtain_dataloader = DataLoader(obtain_dataset, batch_size=256, shuffle=False, drop_last=False,
                                   collate_fn=collate_fn)

    unlearn_result = blindspot_unlearner(amodel, random_model, train_forget_dataloader, forget_dataloader,
                                         obtain_dataloader, test_dataloader, epochs, optimizer,
                                         weights, class_num, bestsavename, lastname, device)
    caculate_conf(amodel, "2分类data9", "Before Forgetting", forget_dataloader, obtain_dataloader, test_dataloader,
                  class_num, device)
