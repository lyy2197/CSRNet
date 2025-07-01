import pickle
import argparse
import time
import torch
from datetime import datetime
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from sklearn.metrics import precision_score, recall_score, f1_score,roc_auc_score
from mydataset import MyDataset
from sklearn.model_selection import train_test_split
from model import net
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from CSRNet import *


def Parser():
    '''
        设置超参数
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--lr', default=0.001, type=float, help="Initial learning rate")
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lam_regularize', default=0.0, type=float, help="The coefficient for the regularizers")
    parser.add_argument('--patience', default=3, type=int, help="早停界限")
    parser.add_argument('--save_to_file', default=False, action='store_true', help="Save results to file")
    args = parser.parse_args([])
    print("训练批次:", args.epochs)
    print("学习率:", args.lr)
    print("最小批次大小:", args.batch_size)
    print("L2正则化系数:", args.lam_regularize)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("当前时间:", current_time)

    return args


def read_data():
    dataset_dir = './test_merge_norm/'
    data_file = 'dataset.pkl'
    label_file = 'labels.pkl'

    with open(dataset_dir + data_file, "rb") as fp:
        datasets = pickle.load(fp)
    with open(dataset_dir + label_file, "rb") as fp:
        labels = pickle.load(fp)

    return datasets,labels


def data_process(datas, labels):
    datas = np.expand_dims(datas, axis=1)

    eeg_train, eeg_test, y_train, y_test = train_test_split(datas, labels, train_size=0.8, random_state=42,
                                                        stratify=labels,shuffle=True)
    eeg_train, eeg_val, y_train, y_val = train_test_split(eeg_train, y_train, train_size=0.9, random_state=42,
                                                      stratify=y_train,shuffle=True)
    print("eeg_train.shape:", eeg_train.shape, "\t y_train.shape:", y_train.shape, )
    print("eeg_val.shape:", eeg_val.shape, "\t y_val.shape:", y_val.shape)
    print("eeg_test.shape:", eeg_test.shape, "\t y_test.shape:", y_test.shape)
    print("------------------------------")

    train_dataset = MyDataset(eeg_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataset = MyDataset(eeg_val, y_val)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataset = MyDataset(eeg_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    return train_dataloader, valid_dataloader, test_dataloader


def train(dataloader, model, loss_fn, optimizer):
    train_loss = 0.0
    size = len(dataloader.dataset)
    model.train()
    correct = 0
    total = 0
    label_len = 0
    all_preds = []
    all_labels = []

    for batch, (eeg_data, label) in enumerate(dataloader, 0):
        eeg_data = eeg_data.to(device)
        label = label.long().to(device)
        label_len = label_len+len(label)
        pred = model(eeg_data)
        _, predicted = torch.max(pred, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(label.cpu().numpy())
        correct += (predicted == label).sum().item()
        loss = loss_fn(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_acc = correct/label_len
    train_loss = train_loss / len(dataloader)
    train_f1 = f1_score(all_labels, all_preds, average='weighted')
    return train_loss, train_acc, train_f1

def predict(dataloader, net, loss_fn):
    net.eval()
    label_len = 0
    correct = 0
    valid_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(dataloader):
            x_batch = x_batch.to(device)
            label = y_batch.long().to(device)
            label_len = label_len + len(label)
            pred = model(x_batch)
            _, predicted = torch.max(pred, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            correct += (predicted == label).sum().item()
            loss = loss_fn(pred, label)
            valid_loss += loss.item()

        valid_acc = correct / label_len
        valid_loss = valid_loss / len(dataloader)

        valid_f1 = f1_score(all_labels, all_preds, average='weighted')

        return valid_loss,valid_acc,valid_f1

def test(test_dataloader, model):
    model.eval()  # 将模型设置为评估模式

    correct = 0
    total = 0
    all_preds = []
    all_labels = []


    with torch.no_grad():
        for data in test_dataloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # 取输出概率最大的类别作为预测结果
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()



    accuracy = correct / total
    test_f1 = f1_score(all_labels, all_preds, average='weighted')
    test_precision = precision_score(all_labels, all_preds, average='weighted')  # or 'macro', 'binary'
    test_recall = recall_score(all_labels, all_preds, average='weighted')  # sensitivity = recall

    print(f'Test Accuracy : {accuracy * 100:.2f}%')
    print(f'Test F1 : {test_f1 * 100:.2f}%')
    print(f'Test Precision : {test_precision * 100:.2f}%')
    print(f'Test Recall : {test_recall * 100:.2f}%')
    # print("------------------------------")
    return accuracy,test_f1,test_precision,test_recall
def load_data(state):
    with open("eeg_" + state + ".pkl", "rb") as f:
        eeg = pickle.load(f)
    with open("y_" + state + ".pkl", "rb") as f:
        y = pickle.load(f)
    return eeg, y



if __name__ == '__main__':
    datas, labels = read_data()

    args = Parser()
    current_file_abs_path = os.getcwd()

    train_dataloader, valid_dataloader, test_dataloader = data_process(datas, labels)
    model = CSRNet()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lam_regularize)
    E_loss = nn.CrossEntropyLoss()

    # 训练和评估模型
    for t in range(args.epochs):
        # print(f"Epoch {t}\n-------------------------------")
        # print('')
        train_loss, train_acc, train_f1 = train(train_dataloader, model, E_loss, optimizer)
        # print(f'epoch {t} epoch_loss {train_loss}\n-------------------------------')
        print('epoch {},train, loss={:.4f} acc={:.4f}  f1={:.4f}'
              .format(t, train_loss, train_acc, train_f1))

        valid_loss, valid_acc,valid_f1 = predict(valid_dataloader, model, E_loss)
        print('epoch {}, val, loss={:.4f} acc={:.4f} f1={:.4f}'
              .format(t, valid_loss, valid_acc, valid_f1))
        print('-------------------------------')

    accuracy,test_f1,test_precision,test_recall = test(test_dataloader, model)






