import os
import torch
import numpy as np
import argparse
import random
import yaml
from easydict import EasyDict
import gensim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.optim as optim

import data_helpers
from models.standard import *

parser = argparse.ArgumentParser(description='PyTorch for image-user CNN')

parser.add_argument('--config', default='config.yaml')
parser.add_argument('--resume', default='', type=str, help='path to checkpoint')  # 增加属性


class ImageFolder(data.Dataset):
    def __init__(self, home_data, work_data, school_data, restaurant_data,
                 shopping_data, cinema_data, sports_data, travel_data):
        _, _, self.x_user, self.y = data_helpers.load_data_and_labels(home_data, work_data,
                                                                      school_data, restaurant_data,
                                                                      shopping_data, cinema_data,
                                                                      sports_data, travel_data)
        self.x_user = torch.Tensor(np.array(self.x_user, dtype="float64"))
        self.y = torch.Tensor(np.array(self.y, dtype="float64"))

    def __getitem__(self, index):
        return self.x_user[index], self.y[index]

    def __len__(self):
        return len(self.x_user)


def cal_acc(pred_user, y):
    pred_user = (pred_user.numpy() == pred_user.numpy().max(axis=1, keepdims=1)).astype("float64")
    pred_user = [np.argmax(item) for item in pred_user]
    y = [np.argmax(item) for item in y]
    pred_user, y = np.array(pred_user), np.array(y)
    per_user = pred_user == y
    user_acc = len(per_user[per_user == True]) / len(per_user) * 100

    return user_acc


def save_state(state, path, epoch):
    print("=> saving checkpoint of epoch " + str(epoch))
    torch.save(state, path + 'params_' + str(epoch) + '.pth')


def load_state(path, netU, optimizerU):
    if not os.path.isfile(path):
        print("=> no checkpoint found at '{}'".format(path))
    else:
        print("=> loading checkpoint '{}'".format(path))

        checkpoint = torch.load(path)
        netU.load_state_dict(checkpoint['state_dictU'])
        optimizerU.load_state_dict(checkpoint['optimizerU'])
        epoch = checkpoint['epoch'] + 1

        return epoch


def main():
    global args, config

    args = parser.parse_args()

    with open(args.config) as f:
        config = EasyDict(yaml.load(f))

    # assert torch.cuda.is_available()
    # device = torch.device("cuda")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # random seed setup
    print("Random Seed: ", config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    cudnn.benchmark = True

    netU = BottlenetU()
    criterion = nn.CrossEntropyLoss()

    netU = netU.to(device)
    criterion = criterion.to(device)

    optimizerU = optim.Adam(netU.parameters(), lr=config.lr_user)

    last_epoch = 0
    if args.resume:
        last_epoch = load_state(args.resume, netU, optimizerU, )

    train_dataset = ImageFolder(config.train.home_data, config.train.work_data, config.train.school_data,
                                config.train.restaurant_data, config.train.shopping_data, config.train.cinema_data,
                                config.train.sports_data, config.train.travel_data)
    train_dataloader = data.DataLoader(train_dataset, batch_size=config.batch_size,
                                       shuffle=True, pin_memory=True, num_workers=int(config.workers))

    for epoch in range(last_epoch, config.epoch - 1):
        for iter, [user, y] in enumerate(train_dataloader):
            # print("epoch: ", epoch, "iter: ", iter)
            # print(netU(user).shape)
            # print(y.shape)
            netU.zero_grad()
            user, y = user.to(device), y.to(device)
            pred_user = netU(user)
            err_user = criterion(pred_user, y.argmax(dim=1))
            # print("err_user: ", err_user)
            err_user.backward()
            optimizerU.step()

        if (epoch + 1) % config.val_freq == 0:
            val(netU.eval(), device)

        if (epoch + 1) % config.save_freq == 0:
            save_state({'state_dictU': netU.state_dict(),
                        'optimizerU': optimizerU.state_dict(),
                        'epoch': epoch}, config.user_save_path, epoch)


def val(netU, device):
    _, _, tr_user, tr_y = data_helpers.load_data_and_labels(config.train.home_data, config.train.work_data,
                                                            config.train.school_data,
                                                            config.train.restaurant_data,
                                                            config.train.shopping_data,
                                                            config.train.cinema_data,
                                                            config.train.sports_data,
                                                            config.train.travel_data)
    _, _, val_user, val_y = data_helpers.load_data_and_labels(config.val.home_data, config.val.work_data,
                                                              config.val.school_data,
                                                              config.val.restaurant_data,
                                                              config.val.shopping_data,
                                                              config.val.cinema_data,
                                                              config.val.sports_data,
                                                              config.val.travel_data)

    tr_user = torch.Tensor(np.array(tr_user, dtype="float64"))
    tr_y = np.array(tr_y, dtype="float64")
    val_user = torch.Tensor(np.array(val_user, dtype="float64"))
    val_y = np.array(val_y, dtype="float64")

    tr_user = tr_user.to(device)
    val_user = val_user.to(device)

    with torch.no_grad():
        pred_tr_user = netU(tr_user)
        pred_val_user = netU(val_user)

    tr_user_acc = cal_acc(pred_tr_user.cpu(), tr_y)
    val_user_acc = cal_acc(pred_val_user.cpu(), val_y)

    print("user accuracy  | train: %.3f %% | test: %.3f %% " % (tr_user_acc, val_user_acc))


if __name__ == "__main__":
    main()
