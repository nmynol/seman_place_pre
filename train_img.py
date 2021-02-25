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
        _, self.x_image, _, self.y = data_helpers.load_data_and_labels(home_data, work_data,
                                                                       school_data, restaurant_data,
                                                                       shopping_data, cinema_data,
                                                                       sports_data, travel_data)

        self.x_image = torch.Tensor(np.array(self.x_image, dtype="float64"))
        self.y = torch.Tensor(np.array(self.y, dtype="float64"))

    def __getitem__(self, index):
        return self.x_image[index], self.y[index]

    def __len__(self):
        return len(self.x_image)


def cal_acc(pred_img, y):
    pred_img = (pred_img.numpy() == pred_img.numpy().max(axis=1, keepdims=1)).astype("float64")
    pred_img = [np.argmax(item) for item in pred_img]
    y = [np.argmax(item) for item in y]
    pred_img, y = np.array(pred_img), np.array(y)
    per_img = pred_img == y
    image_acc = len(per_img[per_img == True]) / len(per_img) * 100

    return image_acc


def save_state(state, path, epoch):
    print("=> saving checkpoint of epoch " + str(epoch))
    torch.save(state, path + 'params_' + str(epoch) + '.pth')


def load_state(path, netI, optimizerI):
    if not os.path.isfile(path):
        print("=> no checkpoint found at '{}'".format(path))
    else:
        print("=> loading checkpoint '{}'".format(path))

        checkpoint = torch.load(path)
        netI.load_state_dict(checkpoint['state_dictI'])
        optimizerI.load_state_dict(checkpoint['optimizerI'])
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

    netI = BottlenetI()
    criterion = nn.CrossEntropyLoss()
    netI = netI.to(device)
    criterion = criterion.to(device)

    optimizerI = optim.Adam(netI.parameters(), lr=config.lr_img)

    last_epoch = 0
    if args.resume:
        last_epoch = load_state(args.resume, netI, optimizerI)

    train_dataset = ImageFolder(config.train.home_data, config.train.work_data, config.train.school_data,
                                config.train.restaurant_data, config.train.shopping_data, config.train.cinema_data,
                                config.train.sports_data, config.train.travel_data)
    train_dataloader = data.DataLoader(train_dataset, batch_size=config.batch_size,
                                       shuffle=True, pin_memory=True, num_workers=int(config.workers))

    for epoch in range(last_epoch, config.epoch - 1):
        for iter, [image, y] in enumerate(train_dataloader):
            # print("epoch: ", epoch, "iter: ", iter)
            # print(netI(image).shape)
            # print(y.shape)
            netI.zero_grad()
            image, y = image.to(device), y.to(device)
            pred_img = netI(image)
            err_img = criterion(pred_img, y.argmax(dim=1))
            # print("err_img: ", err_img)
            err_img.backward()
            optimizerI.step()

        if (epoch + 1) % config.val_freq == 0:
            val(netI.eval(), device)

        if (epoch + 1) % config.save_freq == 0:
            save_state({'state_dictI': netI.state_dict(),
                        'optimizerI': optimizerI.state_dict(),
                        'epoch': epoch}, config.img_save_path, epoch)


def val(netI, device):
    _, tr_image, _, tr_y = data_helpers.load_data_and_labels(config.train.home_data, config.train.work_data,
                                                             config.train.school_data,
                                                             config.train.restaurant_data,
                                                             config.train.shopping_data,
                                                             config.train.cinema_data,
                                                             config.train.sports_data,
                                                             config.train.travel_data)
    _, val_image, _, val_y = data_helpers.load_data_and_labels(config.val.home_data, config.val.work_data,
                                                               config.val.school_data,
                                                               config.val.restaurant_data,
                                                               config.val.shopping_data,
                                                               config.val.cinema_data,
                                                               config.val.sports_data,
                                                               config.val.travel_data)

    tr_image = torch.Tensor(np.array(tr_image, dtype="float64"))
    tr_y = np.array(tr_y, dtype="float64")
    val_image = torch.Tensor(np.array(val_image, dtype="float64"))
    val_y = np.array(val_y, dtype="float64")

    tr_image, val_image = tr_image.to(device), val_image.to(device)

    with torch.no_grad():
        pred_tr_img = netI(tr_image)
        pred_val_img = netI(val_image)

    tr_image_acc = cal_acc(pred_tr_img.cpu(), tr_y)
    val_image_acc = cal_acc(pred_val_img.cpu(), val_y)

    print("image accuracy | train: %.3f %% | test: %.3f %% " % (tr_image_acc, val_image_acc))


if __name__ == "__main__":
    main()
