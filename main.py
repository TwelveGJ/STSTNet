from os import path
import os
import numpy as np
import cv2
import time
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torchsummary import summary
from torch.utils.data import TensorDataset, DataLoader
import argparse
from distutils.util import strtobool
import logging
import random


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


class se_block(nn.Module):
    def __init__(self, channel, ratio=8):
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class STSTNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, use_se_block=False):
        super(STSTNet, self).__init__()
        self.use_se_block = use_se_block
        self.conv1 = nn.Conv2d(in_channels, out_channels=3, kernel_size=3, padding=2)
        # self.se1 = se_block(channel=3)
        self.conv2 = nn.Conv2d(in_channels, out_channels=5, kernel_size=3, padding=2)
        # self.se2 = se_block(channel=5)
        self.conv3 = nn.Conv2d(in_channels, out_channels=8, kernel_size=3, padding=2)
        # self.se3 = se_block(channel=8)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(3)
        self.bn2 = nn.BatchNorm2d(5)
        self.bn3 = nn.BatchNorm2d(8)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=3, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(in_features=5 * 5 * 16, out_features=out_channels)

        if use_se_block:
            self.se1 = se_block(channel=3)
            self.se2 = se_block(channel=5)
            self.se3 = se_block(channel=8)

    def forward(self, x):
        x1 = self.conv1(x)
        if self.use_se_block:
            x1 = self.se1(x1)
        x1 = self.relu(x1)
        x1 = self.bn1(x1)
        x1 = self.maxpool(x1)
        x1 = self.dropout(x1)
        x2 = self.conv2(x)
        if self.use_se_block:
            x2 = self.se2(x2)
        x2 = self.relu(x2)
        x2 = self.bn2(x2)
        x2 = self.maxpool(x2)
        x2 = self.dropout(x2)
        x3 = self.conv3(x)
        if self.use_se_block:
            x3 = self.se3(x3)
        x3 = self.relu(x3)
        x3 = self.bn3(x3)
        x3 = self.maxpool(x3)
        x3 = self.dropout(x3)
        x = torch.cat((x1, x2, x3), 1)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def reset_weights(m):  # Reset the weights for network to avoid weight leakage
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            #             print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


def confusionMatrix(gt, pred, show=False):
    TN, FP, FN, TP = confusion_matrix(gt, pred).ravel()
    f1_score = (2 * TP) / (2 * TP + FP + FN)
    num_samples = len([x for x in gt if x == 1])
    average_recall = TP / num_samples
    return f1_score, average_recall


def recognition_evaluation(final_gt, final_pred, show=False):
    label_dict = {'negative': 0, 'positive': 1, 'surprise': 2}

    # Display recognition result
    f1_list = []
    ar_list = []
    try:
        for emotion, emotion_index in label_dict.items():
            gt_recog = [1 if x == emotion_index else 0 for x in final_gt]
            pred_recog = [1 if x == emotion_index else 0 for x in final_pred]
            try:
                f1_recog, ar_recog = confusionMatrix(gt_recog, pred_recog)
                f1_list.append(f1_recog)
                ar_list.append(ar_recog)
            except Exception as e:
                pass
        UF1 = np.mean(f1_list)
        UAR = np.mean(ar_list)
        return UF1, UAR
    except:
        return '', ''


def main(config):
    seed_everything(config.seed)
    learning_rate = config.lr
    batch_size = config.bs
    epochs = config.epochs

    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    loss_fn = nn.CrossEntropyLoss()

    if config.train:
        if not path.exists('STSTNet_Weights'):
            os.mkdir('STSTNet_Weights')

    logger_file = r'.\logger\logger'
    if config.use_se_block:
        logger_file = r'.\logger\logger_se'
    logging.basicConfig(filename=logger_file + '_seed_' + str(config.seed) + '.txt', level=logging.DEBUG)

    logging.info('train=%d, use_se_block=%d, seed=%d' % (config.train, config.use_se_block, config.seed))
    logging.info('lr=%f, epochs=%d, device=%s' % (learning_rate, epochs, device))
    print('lr=%f, epochs=%d, device=%s\n' % (learning_rate, epochs, device))

    total_gt = []
    total_pred = []

    t = time.time()

    main_path = r'.\input\norm_u_v_os'
    subName = os.listdir(main_path)

    # .\norm_u_v_os
    # |---006
    # |---007
    # |...
    for n_subName in subName:
        logging.info('Subject:' + str(n_subName))
        print('Subject:', n_subName)

        X_train = []
        y_train = []
        X_test = []
        y_test = []

        # Get train dataset
        # .\006_u_train
        # |---0
        # |---1
        # |---2
        expression = os.listdir(main_path + '/' + n_subName + '/u_train')
        for n_expression in expression:
            img = os.listdir(main_path + '/' + n_subName + '/u_train/' + n_expression)

            # .\006_u_train
            # |---0
            # |   |---xxx.png
            # |   |...
            # |...
            for n_img in img:
                y_train.append(int(n_expression))
                X_train.append(cv2.imread(main_path + '/' + n_subName + '/u_train/' + n_expression + '/' + n_img))

        # Get test dataset
        expression = os.listdir(main_path + '/' + n_subName + '/u_test')
        for n_expression in expression:
            img = os.listdir(main_path + '/' + n_subName + '/u_test/' + n_expression)

            for n_img in img:
                y_test.append(int(n_expression))
                X_test.append(cv2.imread(main_path + '/' + n_subName + '/u_test/' + n_expression + '/' + n_img))

        weight_path = 'STSTNet_Weights' + '/' + n_subName + '.pth'

        # Reset or load model weigts
        model = STSTNet(use_se_block=config.use_se_block).to(device)
        if config.train:
            model.apply(reset_weights)
        else:
            model.load_state_dict(torch.load(weight_path))

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Initialize training dataloader
        X_train = torch.Tensor(X_train).permute(0, 3, 1, 2)
        y_train = torch.Tensor(y_train).to(dtype=torch.long)
        dataset_train = TensorDataset(X_train, y_train)
        train_dl = DataLoader(dataset_train, batch_size=batch_size)

        # Initialize testing dataloader
        X_test = torch.Tensor(X_test).permute(0, 3, 1, 2)
        y_test = torch.Tensor(y_test).to(dtype=torch.long)
        dataset_test = TensorDataset(X_test, y_test)
        test_dl = DataLoader(dataset_test, batch_size=batch_size)

        for epoch in range(1, epochs + 1):
            if config.train:
                # Training
                model.train()
                train_loss = 0.0
                num_train_correct = 0
                num_train_examples = 0

                for batch in train_dl:
                    optimizer.zero_grad()
                    x = batch[0].to(device)
                    y = batch[1].to(device)
                    yhat = model(x)
                    loss = loss_fn(yhat, y)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.data.item() * x.size(0)
                    num_train_correct += (torch.max(yhat, 1)[1] == y).sum().item()
                    num_train_examples += x.shape[0]

                train_acc = num_train_correct / num_train_examples
                train_loss = train_loss / len(train_dl.dataset)

            # Testing
            model.eval()
            val_loss = 0.0
            num_val_correct = 0
            num_val_examples = 0
            for batch in test_dl:
                x = batch[0].to(device)
                y = batch[1].to(device)
                yhat = model(x)
                loss = loss_fn(yhat, y)

                val_loss += loss.data.item() * x.size(0)
                num_val_correct += (torch.max(yhat, 1)[1] == y).sum().item()
                num_val_examples += y.shape[0]

            val_acc = num_val_correct / num_val_examples
            val_loss = val_loss / len(test_dl.dataset)
            # print('epoch ' + str(epoch) + 'DONE\n')

            # if epoch == 1 or epoch % 50 == 0:
            #     print('Epoch %3d/%3d, train loss: %5.4f, train acc: %5.4f, val loss: %5.4f, val acc: %5.4f' % (epoch, epochs, train_loss, train_acc, val_loss, val_acc))

        # Save Weights
        if config.train:
            torch.save(model.state_dict(), weight_path)

        # For UF1 and UAR computation
        logging.info('Predicted    :' + str(torch.max(yhat, 1)[1].tolist()))
        logging.info('Ground Truth :' + str(y.tolist()))
        # logging.info('Evaluation until this subject: ')
        print('Predicted    :', torch.max(yhat, 1)[1].tolist())
        print('Ground Truth :', y.tolist())
        print('Evaluation until this subject: ')
        total_pred.extend(torch.max(yhat, 1)[1].tolist())
        total_gt.extend(y.tolist())
        UF1, UAR = recognition_evaluation(total_gt, total_pred, show=True)
        logging.info('UF1:' + str(round(UF1, 4)) + '| UAR:' + str(round(UAR, 4)))
        print('UF1:', round(UF1, 4), '| UAR:', round(UAR, 4))

    print('Final Evaluation: ')
    UF1, UAR = recognition_evaluation(total_gt, total_pred)
    logging.info('UF1:' + str(round(UF1, 4)) + '| UAR:' + str(round(UAR, 4)))
    logging.info('Total Time Taken:' + str(time.time() - t))
    print('UF1:', round(UF1, 4), '| UAR:', round(UAR, 4))
    print('Total Time Taken:', time.time() - t)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input parameters
    parser.add_argument('--train', type=bool, default=False)  # Train or use pre-trained weight for prediction
    parser.add_argument('--use_se_block', type=bool, default=False)
    # parser.add_argument('--logger', type=str, default=r'./logger.txt')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--bs', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=800)
    config = parser.parse_args()
    main(config)
