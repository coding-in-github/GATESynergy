import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch.nn as nn
from model import GATESynergy
from dataset import TestbedDataset
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score, \
    balanced_accuracy_score, f1_score
from sklearn import metrics
from process_data import process_data
import datetime
import torch
import random
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def create_cv_splits(length, n_folds=5, seed=42):
    np.random.seed(seed)
    indices = np.random.permutation(length)
    return indices


def save_AUCs(AUCs, filename):
    with open(filename, 'a') as f:
        f.write('\t'.join(map(str, AUCs)) + '\n')


def train(model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch):
    model.train()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()
    total_loss = 0.0
    num_batches = 0

    zipped = zip(drug1_loader_train, drug2_loader_train)
    enumerate_data = enumerate(zipped)
    for batch_idx, data in enumerate_data:
        data1 = data[0]
        data2 = data[1]
        data1 = data1.to(device)
        data2 = data2.to(device)
        y = data[0].y.view(-1, 1).float().to(device)
        y = y.squeeze(1)
        optimizer.zero_grad()
        output = model(data1, data2)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
        ys = output.to('cpu').data.numpy()
        predicted_labels = list(map(lambda x: int(x > 0.5), ys))
        predicted_scores = list(map(lambda x: x, ys))
        total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
        total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
        total_labels = torch.cat((total_labels, data1.y.view(-1, 1).cpu()), 0)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_prelabels.numpy().flatten(), avg_loss


def predicting(model, device, drug1_loader_test, drug2_loader_test, loss_fn):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for data in zip(drug1_loader_test, drug2_loader_test):
            data1 = data[0]
            data2 = data[1]
            data1 = data1.to(device)
            data2 = data2.to(device)
            y = data[0].y.view(-1, 1).float().to(device)
            y = y.squeeze(1)
            output = model(data1, data2)
            loss = loss_fn(output, y)
            total_loss += loss.item()
            num_batches += 1
            ys = output.to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: int(x > 0.5), ys))
            predicted_scores = list(map(lambda x: x, ys))
            total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
            total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
            total_labels = torch.cat((total_labels, data1.y.view(-1, 1).cpu()), 0)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_prelabels.numpy().flatten(), avg_loss


set_seed(42)
device = torch.device('cuda:0')
TRAIN_BATCH_SIZE = 1024
TEST_BATCH_SIZE = 1024
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 300
ETA_MIN = 1e-6

cell_gene_expressions = 'data/cell_gene_expressions.csv'
drug_smiles = 'data/drug_smiles.csv'
drug_synergy_dataset = 'data/drug_synergy_dataset.csv'
dataset_name = 'drug_synergy_dataset'

drug1, drug2, cell, label, smile_graph, cell_features = process_data(drug_synergy_dataset, drug_smiles,
                                                                   cell_gene_expressions)
drug1_data = TestbedDataset(dataset=dataset_name + '_drug1', xd=drug1, xt=cell, y=label, smile_graph=smile_graph,
                            xt_featrue=cell_features)
drug2_data = TestbedDataset(dataset=dataset_name + '_drug2', xd=drug2, xt=cell, y=label, smile_graph=smile_graph,
                            xt_featrue=cell_features)

result_name = 'GATESynergy'
modeling = GATESynergy
lenth = len(drug1_data)
pot = int(lenth / 5)

random_num = create_cv_splits(lenth, n_folds=5, seed=42).tolist()

folder_path = './result/' + result_name
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

all_fold_metrics = {
    'AUC': [],
    'PR_AUC': [],
    'ACC': [],
    'BACC': [],
    'PREC': [],
    'TPR': [],
    'KAPPA': [],
    'RECALL': [],
    'F1': []
}
all_fold_best_results = []

print("=" * 60)
print("Starting 5-Fold Cross-Validation Training")
print("=" * 60)

for i in range(5):
    print(f"\n{'=' * 20} Starting Training for Fold {i + 1} {'=' * 20}")
    test_num = random_num[pot * i:pot * (i + 1)]
    train_num = random_num[:pot * i] + random_num[pot * (i + 1):]

    drug1_data_train = drug1_data[train_num]
    drug1_data_test = drug1_data[test_num]
    drug1_loader_train = DataLoader(drug1_data_train, batch_size=TRAIN_BATCH_SIZE, shuffle=None)
    drug1_loader_test = DataLoader(drug1_data_test, batch_size=TRAIN_BATCH_SIZE, shuffle=None)

    drug2_data_test = drug2_data[test_num]
    drug2_data_train = drug2_data[train_num]
    drug2_loader_train = DataLoader(drug2_data_train, batch_size=TRAIN_BATCH_SIZE, shuffle=None)
    drug2_loader_test = DataLoader(drug2_data_test, batch_size=TRAIN_BATCH_SIZE, shuffle=None)

    print("===============")
    model = modeling().to(device)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=ETA_MIN)

    now = datetime.datetime.now()
    time_str = now.strftime("%Y-%m-%d %H_%M_%S")

    file_AUCs = folder_path + '/' + result_name + '_fold_' + str(
        i + 1) + '--AUCs--' + dataset_name + '_' + time_str + '.txt'
    AUCs = ('Epoch\tLR\tAUC_dev\tPR_AUC\tACC\tBACC\tPREC\tTPR\tKAPPA\tRECALL\tF1')
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')

    best_auc = 0
    best_metrics = None
    best_epoch = 0

    for epoch in range(NUM_EPOCHS):
        current_lr = optimizer.param_groups[0]['lr']
        train_T, train_S, train_Y, train_loss = train(model, device, drug1_loader_train, drug2_loader_train, optimizer,
                                                      epoch + 1)

        T, S, Y, test_loss = predicting(model, device, drug1_loader_test, drug2_loader_test, loss_fn)
        AUC = roc_auc_score(T, S)
        precision, recall, threshold = metrics.precision_recall_curve(T, S)
        PR_AUC = metrics.auc(recall, precision)
        BACC = balanced_accuracy_score(T, Y)
        tn, fp, fn, tp = confusion_matrix(T, Y).ravel()
        TPR = tp / (tp + fn)
        PREC = precision_score(T, Y)
        ACC = accuracy_score(T, Y)
        KAPPA = cohen_kappa_score(T, Y)
        recall_score_val = recall_score(T, Y)
        F1 = f1_score(T, Y)

        train_AUC = roc_auc_score(train_T, train_S)
        train_precision_curve, train_recall_curve, train_threshold = metrics.precision_recall_curve(train_T, train_S)
        train_PR_AUC = metrics.auc(train_recall_curve, train_precision_curve)
        train_BACC = balanced_accuracy_score(train_T, train_Y)
        train_tn, train_fp, train_fn, train_tp = confusion_matrix(train_T, train_Y).ravel()
        train_TPR = train_tp / (train_tp + train_fn)
        train_PREC = precision_score(train_T, train_Y)
        train_ACC = accuracy_score(train_T, train_Y)
        train_KAPPA = cohen_kappa_score(train_T, train_Y)
        train_recall_score_val = recall_score(train_T, train_Y)
        train_F1 = f1_score(train_T, train_Y)

        if best_auc < AUC:
            best_auc = AUC
            best_epoch = epoch + 1
            best_metrics = {
                'AUC': AUC,
                'PR_AUC': PR_AUC,
                'ACC': ACC,
                'BACC': BACC,
                'PREC': PREC,
                'TPR': TPR,
                'KAPPA': KAPPA,
                'RECALL': recall_score_val,
                'F1': F1
            }

            AUCs = [epoch, current_lr, AUC, PR_AUC, ACC, BACC, PREC, TPR, KAPPA, recall_score_val, F1]
            save_AUCs(AUCs, file_AUCs)

        if (epoch + 1) % 10 == 0:
            print(
                f"Fold {i + 1} - Epoch {epoch + 1}:  LR: {current_lr:.2e}, Training Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
            print(
                f"  Train: AUC={train_AUC:.4f}, PR_AUC={train_PR_AUC:.4f}, ACC={train_ACC:.4f}, BACC={train_BACC:.4f}, PREC={train_PREC:.4f}, TPR={train_TPR:.4f}, KAPPA={train_KAPPA:.4f}, RECALL={train_recall_score_val:.4f}, F1={train_F1:.4f}")
            print(
                f"  Test:  AUC={AUC:.4f}, PR_AUC={PR_AUC:.4f}, ACC={ACC:.4f}, BACC={BACC:.4f}, PREC={PREC:.4f}, TPR={TPR:.4f}, KAPPA={KAPPA:.4f}, RECALL={recall_score_val:.4f}, F1={F1:.4f}")
            print(
                f"  Current Best: Epoch {best_epoch}, AUC={best_metrics['AUC']:.4f}, PR_AUC={best_metrics['PR_AUC']:.4f}, ACC={best_metrics['ACC']:.4f}, BACC={best_metrics['BACC']:.4f}, PREC={best_metrics['PREC']:.4f}, TPR={best_metrics['TPR']:.4f}, KAPPA={best_metrics['KAPPA']:.4f}, RECALL={best_metrics['RECALL']:.4f}, F1={best_metrics['F1']:.4f}")

        scheduler.step()

    for metric_name, metric_value in best_metrics.items():
        all_fold_metrics[metric_name].append(metric_value)

    fold_best_result = {
        'fold': i + 1,
        'epoch': best_epoch,
        'metrics': best_metrics.copy()
    }
    all_fold_best_results.append(fold_best_result)

    save_AUCs("best_auc:" + str(best_auc), file_AUCs)
    print(f"\nFold {i + 1} training completed. Best result at epoch {best_epoch}, AUC: {best_auc:.4f}")

print("\n" + "=" * 60)
print("Best Results Summary for Each Fold")
print("=" * 60)
for fold_result in all_fold_best_results:
    fold_num = fold_result['fold']
    epoch_num = fold_result['epoch']
    metrics = fold_result['metrics']
    metrics_str = ", ".join([f"{name}={value:.4f}" for name, value in metrics.items()])
    print(f"Fold {fold_num} Best Result (Epoch {epoch_num}):    {metrics_str}")

print("\n" + "=" * 60)
print("Average Results of 5-Fold Cross Validation")
print("=" * 60)

summary_results = []
for metric_name in all_fold_metrics.keys():
    values = all_fold_metrics[metric_name]
    mean_val = np.mean(values)
    std_val = np.std(values)
    print(f"{metric_name}: {mean_val:.4f} ± {std_val:.4f}")
    summary_results.append(f"{metric_name}: {mean_val:.4f} ± {std_val:.4f}")

now = datetime.datetime.now()
time_str = now.strftime("%Y-%m-%d %H_%M_%S")
summary_file = folder_path + '/CV_Summary_' + result_name + '_' + dataset_name + '_' + time_str + '.txt'

with open(summary_file, 'w') as f:
    f.write("5-Fold Cross Validation Summary\n")
    f.write("=" * 50 + "\n")
    f.write(f"Dataset: {dataset_name}\n")
    f.write(f"Model: {result_name}\n")
    f.write(f"Initial LR: {LR}\n")
    f.write(f"Min LR (eta_min): {ETA_MIN}\n")
    f.write(f"Scheduler: CosineAnnealingLR\n")
    f.write(f"Training completed at: {time_str}\n\n")

    f.write("Individual Fold Best Results:\n")
    for fold_result in all_fold_best_results:
        fold_num = fold_result['fold']
        epoch_num = fold_result['epoch']
        metrics = fold_result['metrics']
        f.write(f"\nFold {fold_num} Best Result (Epoch {epoch_num}):\n")
        for metric_name, metric_value in metrics.items():
            f.write(f"  {metric_name}: {metric_value:.4f}\n")

    f.write("\nOverall Results (Mean ± Std):\n")
    for result_line in summary_results:
        f.write(result_line + "\n")

print(f"\nSummary results have been saved to: {summary_file}")
print("=" * 60)
