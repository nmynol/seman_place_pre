import os
import torch
import numpy as np
import argparse
import random
import yaml
from easydict import EasyDict
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
        _, self.x_image, self.x_user, self.y = data_helpers.load_data_and_labels(home_data, work_data,
                                                                                           school_data, restaurant_data,
                                                                                           shopping_data, cinema_data,
                                                                                           sports_data, travel_data)

        self.x_image = torch.Tensor(np.array(self.x_image, dtype="float64"))
        self.x_user = torch.Tensor(np.array(self.x_user, dtype="float64"))
        self.y = torch.Tensor(np.array(self.y, dtype="float64"))
        self.x_mix = torch.cat((self.x_image, self.x_user), dim=1)

    def __getitem__(self, index):
        return self.x_mix[index], self.y[index]

    def __len__(self):
        return len(self.x_mix)


def cal_acc(pred_mix, y):
    pred_mix = (pred_mix.numpy() == pred_mix.numpy().max(axis=1, keepdims=1)).astype("float64")
    pred_mix = [np.argmax(item) for item in pred_mix]
    y = [np.argmax(item) for item in y]
    pred_mix, y = np.array(pred_mix), np.array(y)
    per_mix = pred_mix == y
    mix_acc = len(per_mix[per_mix == True]) / len(per_mix) * 100

    return mix_acc


def save_state(state, path, epoch):
    print("=> saving checkpoint of epoch " + str(epoch))
    torch.save(state, path + 'params_' + str(epoch) + '.pth')


def load_state(path, netM, optimizerM):
    if not os.path.isfile(path):
        print("=> no checkpoint found at '{}'".format(path))
    else:
        print("=> loading checkpoint '{}'".format(path))

        checkpoint = torch.load(path)
        netM.load_state_dict(checkpoint['state_dictM'])
        optimizerM.load_state_dict(checkpoint['optimizerM'])
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

    netM = BottlenetM()
    criterion = nn.CrossEntropyLoss()
    netM = netM.to(device)
    criterion = criterion.to(device)
    optimizerM = optim.Adam(netM.parameters(), lr=config.lr_mix)

    last_epoch = 0
    if args.resume:
        last_epoch = load_state(args.resume, netM, optimizerM)

    train_dataset = ImageFolder(config.train.home_data, config.train.work_data, config.train.school_data,
                                config.train.restaurant_data, config.train.shopping_data, config.train.cinema_data,
                                config.train.sports_data, config.train.travel_data)
    train_dataloader = data.DataLoader(train_dataset, batch_size=config.batch_size,
                                       shuffle=True, pin_memory=True, num_workers=int(config.workers))

    for epoch in range(last_epoch, config.epoch - 1):
        for iter, [mix, y] in enumerate(train_dataloader):

            netM.zero_grad()
            mix, y = mix.to(device), y.to(device)
            pred_mix = netM(mix)
            err_mix = criterion(pred_mix, y.argmax(dim=1))
            # print("err_mix: ", err_mix)
            err_mix.backward()
            optimizerM.step()

        if (epoch + 1) % config.val_freq == 0:
            val(netM.eval(), device)

        if (epoch) % config.save_freq == 0:
            save_state({'state_dictM': netM.state_dict(),
                        'optimizerM': optimizerM.state_dict(),
                        'epoch': epoch}, config.img_user_save_path, epoch)


def val(netM, device):
    _, tr_image, tr_user, tr_y = data_helpers.load_data_and_labels(config.train.home_data, config.train.work_data,
                                                                             config.train.school_data,
                                                                             config.train.restaurant_data,
                                                                             config.train.shopping_data,
                                                                             config.train.cinema_data,
                                                                             config.train.sports_data,
                                                                             config.train.travel_data)
    _, val_image, val_user, val_y = data_helpers.load_data_and_labels(config.val.home_data, config.val.work_data,
                                                                             config.val.school_data,
                                                                             config.val.restaurant_data,
                                                                             config.val.shopping_data,
                                                                             config.val.cinema_data,
                                                                             config.val.sports_data,
                                                                             config.val.travel_data)

    tr_image = torch.Tensor(np.array(tr_image, dtype="float64"))
    tr_user = torch.Tensor(np.array(tr_user, dtype="float64"))
    tr_mix = torch.cat((tr_image, tr_user), dim=1)
    tr_y = np.array(tr_y, dtype="float64")
    val_image = torch.Tensor(np.array(val_image, dtype="float64"))
    val_user = torch.Tensor(np.array(val_user, dtype="float64"))
    val_mix = torch.cat((val_image, val_user), dim=1)
    val_y = np.array(val_y, dtype="float64")

    tr_mix, val_mix = tr_mix.to(device), val_mix.to(device)

    with torch.no_grad():
        pred_tr_mix = netM(tr_mix)
        pred_val_mix = netM(val_mix)

    tr_mix_acc = cal_acc(pred_tr_mix.cpu(), tr_y)
    val_mix_acc = cal_acc(pred_val_mix.cpu(), val_y)

    print("mix accuracy   | train: %.3f %% | test: %.3f %% " % (tr_mix_acc, val_mix_acc))


if __name__ == "__main__":
    main()
