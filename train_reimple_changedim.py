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
import torch.nn as nn
from tensorboardX import SummaryWriter
from torchvision.transforms import transforms

import data_helpers
from models.origin_model import CNN_fusion


def generate_list(text_raw):
    word_list = []
    word = ""
    for i in text_raw:
        if (i != u" "):
            word = word + i
        else:
            word_list.append(word)
            word = ""
    word_list.append(word)
    return ' '.join(word_list)


def load_text(x_text, name):
    # text_list = []
    # for x in x_text:
    #     x_u = str(x)
    #     xlist = generate_list(x_u)
    #     text_list.append(xlist)
    # max_document_length = max([len(text.split(" ")) for text in text_list])
    # print("max_document_length in " + name + " set:")
    # print(max_document_length)
    #
    # print("using word2vec to embed words in " + name + " set...")
    # model = gensim.models.KeyedVectors.load_word2vec_format('./alldata/vectors300.txt', binary=False)
    # print("embedding completed.")
    # # print(text_list)
    # all_vectors = []
    # embeddingUnknown = [0 for i in range(config.embedding_dim)]

    if not os.path.exists('./alldata/x_text_' + name + '.npy'):
        print("constructing x_text_" + name + "....")
        # for text in text_list:
        #     this_vector = []
        #     text = text.split(" ")
        #     if len(text) < max_document_length:
        #         text.extend(['<PADDING>'] * (max_document_length - len(text)))
        #     for word in text:
        #         if word in model.index2word:
        #             this_vector.append(model[word])
        #         else:
        #             this_vector.append(embeddingUnknown)
        #     all_vectors.append(this_vector)
        #     print(len(all_vectors))
        # x_text = np.array(all_vectors)
        # print("construction completed.")
        # print("saving x_text_ " + name + "...")
        # np.save("./alldata/x_text_" + name + ".npy", x_text)
        # print("x_text saved.")
    else:
        print("loading x_text_" + name + ".....")
        x_text = np.load("./alldata/x_text_" + name + ".npy")
        print("x_text_" + name + " loaded.")
    return x_text


class ImageFolder(data.Dataset):
    def __init__(self, x_text, x_image, x_user, y):
        self.x_text = x_text
        self.x_image = x_image
        self.x_user = x_user
        self.y = y
        self.x_text = torch.Tensor(self.x_text.astype("float64"))
        self.x_image = torch.Tensor(self.x_image.astype("float64"))
        self.x_user = torch.Tensor(self.x_user.astype("float64"))
        self.y = torch.Tensor(np.array(self.y, dtype="float64"))

    def __getitem__(self, index):

        return self.x_text[index], self.x_image[index], self.x_user[index], self.y[index]

    def __len__(self):
        return len(self.x_text)


def show_condition(x_image, x_user):
    x_image_list = []
    for image in x_image:
        image = image.tolist()
        image = [float(i) for i in image]
        x_image_list.append(image)
    x_image = np.array(x_image_list)

    x_user_list = []
    for user in x_user:
        user = user.tolist()
        user = [float(i) for i in user]
        x_user_list.append(user)
    x_user = np.array(x_user_list)

    return x_image, x_user


def data_prepare(config, mode):
    train_text, train_image, train_user, train_y = data_helpers.load_data_and_labels(config.train.home_data,
                                                                                     config.train.work_data,
                                                                                     config.train.school_data,
                                                                                     config.train.restaurant_data,
                                                                                     config.train.shopping_data,
                                                                                     config.train.cinema_data,
                                                                                     config.train.sports_data,
                                                                                     config.train.travel_data)
    test_text, test_image, test_user, test_y = data_helpers.load_data_and_labels(config.val.home_data,
                                                                                 config.val.work_data,
                                                                                 config.val.school_data,
                                                                                 config.val.restaurant_data,
                                                                                 config.val.shopping_data,
                                                                                 config.val.cinema_data,
                                                                                 config.val.sports_data,
                                                                                 config.val.travel_data)
    train_text = load_text(train_text, "train_400")

    train_image, train_user = show_condition(train_image, train_user)
    test_image, test_user = show_condition(test_image, test_user)

    if mode == 'total':
        print('data mode: shuffle')
        # if train_text.shape[1] > test_text.shape[1]:
        #     test_text = np.pad(test_text, (
        #         (0, 0),
        #         (0, max(train_text.shape[1], test_text.shape[1]) - min(train_text.shape[1], test_text.shape[1])),
        #         (0, 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
        # if train_text.shape[1] < test_text.shape[1]:
        #     train_text = np.pad(train_text, (
        #         (0, 0),
        #         (0, max(train_text.shape[1], test_text.shape[1]) - min(train_text.shape[1], test_text.shape[1])),
        #         (0, 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
        text = train_text
        image = np.concatenate((train_image, test_image), axis=0)
        user = np.concatenate((train_user, test_user), axis=0)
        y = np.concatenate((train_y, test_y), axis=0)

        permutation = np.random.permutation(text.shape[0])
        shuffled_text = text[permutation, :, :]
        shuffled_image = image[permutation, :]
        shuffled_user = user[permutation, :]
        shuffled_y = y[permutation, :]
        train_text, test_text, train_image, test_image, train_user, test_user, \
        train_y, test_y = shuffled_text[3000:shuffled_text.shape[0], :, :], shuffled_text[0:3000, :, :], \
                          shuffled_image[3000:shuffled_image.shape[0], :], shuffled_image[0:3000, :], \
                          shuffled_user[3000:shuffled_user.shape[0], :], shuffled_user[0:3000, :], \
                          shuffled_y[3000:shuffled_y.shape[0], :], shuffled_y[0:3000, :]
    print(train_text.shape)
    print(test_text.shape)
    return train_text, test_text, train_image, test_image, train_user, test_user, train_y, test_y


def cal_acc(pred_text, y):
    pred_text = (pred_text.numpy() == pred_text.numpy().max(axis=1, keepdims=1)).astype("float64")
    pred_text = [np.argmax(item) for item in pred_text]
    y = [np.argmax(item) for item in y]
    pred_text, y = np.array(pred_text), np.array(y)
    per_text = pred_text == y
    text_acc = len(per_text[per_text == True]) / len(per_text) * 100

    return text_acc


def save_state(state, path, epoch):
    print("=> saving checkpoint of epoch " + str(epoch))
    torch.save(state, path + 'params_' + str(epoch) + '.pth')


def load_state(path, netF, optimizerF):
    if not os.path.isfile(path):
        print("=> no checkpoint found at '{}'".format(path))
    else:
        print("=> loading checkpoint '{}'".format(path))

        checkpoint = torch.load(path)
        netF.load_state_dict(checkpoint['state_dictF'])
        optimizerF.load_state_dict(checkpoint['optimizerF'])
        epoch = checkpoint['epoch'] + 1
        count = checkpoint['count']

        return epoch, count


def val(tr_text_emb, val_text_emb, tr_image, val_image, tr_user, val_user, tr_y, val_y, netF, device, epoch, writer):
    tr_text = torch.Tensor(np.array(tr_text_emb, dtype="float64"))
    tr_image = torch.Tensor(np.array(tr_image, dtype="float64"))
    tr_user = torch.Tensor(np.array(tr_user, dtype="float64"))
    tr_y = np.array(tr_y, dtype="float64")
    val_text = torch.Tensor(np.array(val_text_emb, dtype="float64"))
    val_image = torch.Tensor(np.array(val_image, dtype="float64"))
    val_user = torch.Tensor(np.array(val_user, dtype="float64"))
    val_y = np.array(val_y, dtype="float64")

    tr_text, tr_image, tr_user = tr_text.to(device), tr_image.to(device), tr_user.to(device)
    val_text, val_image, val_user = val_text.to(device), val_image.to(device), val_user.to(device)

    with torch.no_grad():
        pred_tr = netF(tr_text, tr_image, tr_user)
        pred_val = netF(val_text, val_image, val_user)

    tr_text_acc = cal_acc(pred_tr.cpu(), tr_y)
    val_text_acc = cal_acc(pred_val.cpu(), val_y)

    print("epoch " + str(epoch), " fusion accuracy  | train: %.3f %% | test: %.3f %% " % (tr_text_acc, val_text_acc))
    writer.add_scalar("train_accuracy", tr_text_acc, epoch)
    writer.add_scalar("val_accuracy", val_text_acc, epoch)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch for image-user CNN')

    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--resume', default='', type=str, help='path to checkpoint')  # 增加属性
    parser.add_argument('--data_mode', default='total', type=str)  # 决定用哪个数据集
    parser.add_argument('--model_mode', default='simple', type=str)  # 决定用哪种融合方式
    # parser.add_argument('--board_path', default='./board/', help='')  # 增加属性
    # parser.add_argument('--board_freq', default=10, help='')  # 增加属性
    args = parser.parse_args()

    with open(args.config) as f:
        config = EasyDict(yaml.load(f))

    # assert torch.cuda.is_available()
    # device = torch.device("cuda")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("device:", device)
    # random seed setup
    print("Random Seed: ", config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    cudnn.benchmark = True

    netF = CNN_fusion(args.model_mode)
    criterion = nn.CrossEntropyLoss()
    netF = netF.to(device)
    criterion = criterion.to(device)

    optimizerF = optim.Adam(netF.parameters(), lr=config.lr_fusion)

    last_epoch = 0
    count = 0

    if args.resume:
        last_epoch, count = load_state(args.resume, netF, optimizerF)

    tr_text_emb, test_text_emb, tr_image, test_image, \
    tr_user, test_user, tr_y, test_y = data_prepare(config, args.data_mode)

    train_dataset = ImageFolder(tr_text_emb, tr_image, tr_user, tr_y)
    train_dataloader = data.DataLoader(train_dataset, batch_size=config.batch_size, drop_last=True,
                                       shuffle=True)

    writer = SummaryWriter(config.board_path)

    for epoch in range(last_epoch, config.epoch - 1):
        for iter, [text, image, user, y] in enumerate(train_dataloader):
            netF.train()
            netF.zero_grad()
            text, image, user, y = text.to(device), image.to(device), user.to(device), y.to(device)
            pred = netF(text, image, user)
            err_text = criterion(pred, y.argmax(dim=1))
            # print("err: ", err_text)
            err_text.backward()
            optimizerF.step()
            count = count + 1

            if (iter + 1) % config.board_freq == 0:
                writer.add_scalar("Loss", err_text.item(), count)

        if (epoch + 1) % config.val_freq == 0:
            val(tr_text_emb, test_text_emb, tr_image, test_image, tr_user, test_user, tr_y, test_y, netF.eval(), device,
                epoch, writer)

        if (epoch + 1) % config.save_freq == 0:
            save_state({'state_dictF': netF.state_dict(),
                        'optimizerF': optimizerF.state_dict(),
                        'epoch': epoch,
                        'count': count}, config.fusion_save_path, epoch)
