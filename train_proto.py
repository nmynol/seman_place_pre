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


def load_text(x_text):
    text_list = []
    for x in x_text:
        x_u = str(x)
        xlist = generate_list(x_u)
        text_list.append(xlist)
    max_document_length = max([len(text.split(" ")) for text in text_list])
    print("max_document_length:")
    print(max_document_length)

    print("using word2vec to embed words...")
    model = gensim.models.KeyedVectors.load_word2vec_format('./alldata/vectors300.txt', binary=False)
    print("embedding completed.")
    # print(text_list)
    all_vectors = []
    embeddingUnknown = [0 for i in range(config.embedding_dim)]

    if not os.path.exists('alldata/x_text_train.npy'):
        print("constructing x_text....")
        for text in text_list:
            this_vector = []
            text = text.split(" ")
            if len(text) < max_document_length:
                text.extend(['<PADDING>'] * (max_document_length - len(text)))
            for word in text:
                if word in model.index2word:
                    this_vector.append(model[word])
                else:
                    this_vector.append(embeddingUnknown)
            all_vectors.append(this_vector)
            print(len(all_vectors))
        x_text = np.array(all_vectors)
        print("construction completed.")
        print("saving x_text...")
        np.save("alldata/x_text_train.npy", x_text)
        print("x_text saved.")
    else:
        print("loading x_text.....")
        x_text = np.load("alldata/x_text_train.npy")
        print("x_text loaded.")
    return x_text


class ImageFolder(data.Dataset):
    def __init__(self, home_data, work_data, school_data, restaurant_data,
                 shopping_data, cinema_data, sports_data, travel_data):
        self.x_text, self.x_image, self.x_user, self.y = data_helpers.load_data_and_labels(home_data, work_data,
                                                                                           school_data, restaurant_data,
                                                                                           shopping_data, cinema_data,
                                                                                           sports_data, travel_data)
        self.x_text = torch.Tensor(load_text(self.x_text).astype("float64"))
        self.x_image = torch.Tensor(np.array(self.x_image, dtype="float64"))
        self.x_user = torch.Tensor(np.array(self.x_user, dtype="float64"))
        self.y = torch.Tensor(np.array(self.y, dtype="float64"))
        print(self.x_text.shape)
        print(self.x_image.shape)
        print(self.x_user.shape)
        print(self.y.shape)

    def __getitem__(self, index):
        return self.x_image[index], self.x_user[index], self.y[index]

    def __len__(self):
        return len(self.x_text)


def cal_acc(pred_img, pred_user, pred_mix, y):
    pred_img = (pred_img.numpy() == pred_img.numpy().max(axis=1, keepdims=1)).astype("float64")
    pred_user = (pred_user.numpy() == pred_user.numpy().max(axis=1, keepdims=1)).astype("float64")
    pred_mix = (pred_mix.numpy() == pred_mix.numpy().max(axis=1, keepdims=1)).astype("float64")

    pred_img = [np.argmax(item) for item in pred_img]
    pred_user = [np.argmax(item) for item in pred_user]
    pred_mix = [np.argmax(item) for item in pred_mix]
    y = [np.argmax(item) for item in y]

    pred_img, pred_user, pred_mix, y = np.array(pred_img), np.array(pred_user), np.array(pred_mix), np.array(y)

    per_img = pred_img == y
    per_user = pred_user == y
    per_mix = pred_mix == y

    image_acc = len(per_img[per_img == True]) / len(per_img) * 100
    user_acc = len(per_user[per_user == True]) / len(per_user) * 100
    mix_acc = len(per_mix[per_mix == True]) / len(per_mix) * 100

    return image_acc, user_acc, mix_acc


def save_state(state, path, epoch):
    print("=> saving checkpoint of epoch " + str(epoch))
    torch.save(state, path + 'params_' + str(epoch) + '.pth')


def load_state(path, netI, netU, netM, optimizerI, optimizerU, optimizerM):
    if not os.path.isfile(path):
        print("=> no checkpoint found at '{}'".format(path))
    else:
        print("=> loading checkpoint '{}'".format(path))

        checkpoint = torch.load(path)
        netI.load_state_dict(checkpoint['state_dictI'])
        netU.load_state_dict(checkpoint['state_dictU'])
        netM.load_state_dict(checkpoint['state_dictM'])
        optimizerI.load_state_dict(checkpoint['optimizerI'])
        optimizerU.load_state_dict(checkpoint['optimizerU'])
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

    netI = BottlenetI(205)
    netU = BottlenetU(12)
    netM = BottlenetM(205 + 12)

    criterion = nn.CrossEntropyLoss()

    netI = netI.to(device)
    netU = netU.to(device)
    netM = netM.to(device)
    criterion = criterion.to(device)

    optimizerI = optim.Adam(netI.parameters(), lr=config.lr_img)
    optimizerU = optim.Adam(netU.parameters(), lr=config.lr_user)
    optimizerM = optim.Adam(netM.parameters(), lr=config.lr_mix)

    last_epoch = 0
    if args.resume:
        last_epoch = load_state(args.resume, netI, netU, netM, optimizerI, optimizerU, optimizerM)

    train_dataset = ImageFolder(config.train.home_data, config.train.work_data, config.train.school_data,
                                config.train.restaurant_data, config.train.shopping_data, config.train.cinema_data,
                                config.train.sports_data, config.train.travel_data)
    train_dataloader = data.DataLoader(train_dataset, batch_size=config.batch_size,
                                       shuffle=True, pin_memory=True, num_workers=int(config.workers))

    for epoch in range(last_epoch, config.epoch - 1):
        for iter, [image, user, y] in enumerate(train_dataloader):
            # print("epoch: ", epoch, "iter: ", iter)
            # print(netI(image).shape)
            # print(netU(user).shape)
            # print(y.shape)
            netI.zero_grad()
            netU.zero_grad()
            netM.zero_grad()

            mix = torch.cat((image, user), dim=1)
            image, user, mix, y = image.to(device), user.to(device), mix.to(device), y.to(device)

            pred_img = netI(image)
            pred_user = netU(user)
            pred_mix = netM(mix)

            err_img = criterion(pred_img, y.argmax(dim=1))
            err_user = criterion(pred_user, y.argmax(dim=1))
            err_mix = criterion(pred_mix, y.argmax(dim=1))

            # print("err_img: ", err_img)
            # print("err_user: ", err_user)
            # print("err_mix: ", err_mix)

            err_img.backward()
            err_user.backward()
            err_mix.backward()

            optimizerI.step()
            optimizerU.step()
            optimizerM.step()

        if (epoch + 1) % config.val_freq == 0:
            val(netI.eval(), netU.eval(), netM.eval(), device)

        if (epoch + 1) % config.save_freq == 0:
            save_state({'state_dictI': netI.state_dict(),
                        'state_dictU': netU.state_dict(),
                        'state_dictM': netM.state_dict(),
                        'optimizerI': optimizerI.state_dict(),
                        'optimizerU': optimizerU.state_dict(),
                        'optimizerM': optimizerM.state_dict(),
                        'epoch': epoch}, config.save_path, epoch)


def val(netI, netU, netM, device):
    tr_text, tr_image, tr_user, tr_y = data_helpers.load_data_and_labels(config.train.home_data, config.train.work_data,
                                                                             config.train.school_data,
                                                                             config.train.restaurant_data,
                                                                             config.train.shopping_data,
                                                                             config.train.cinema_data,
                                                                             config.train.sports_data,
                                                                             config.train.travel_data)
    val_text, val_image, val_user, val_y = data_helpers.load_data_and_labels(config.val.home_data, config.val.work_data,
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

    tr_image, tr_user, tr_mix = tr_image.to(device), tr_user.to(device), tr_mix.to(device)
    val_image, val_user, val_mix = val_image.to(device), val_user.to(device), val_mix.to(device)

    with torch.no_grad():
        pred_tr_img = netI(tr_image)
        pred_tr_user = netU(tr_user)
        pred_tr_mix = netM(tr_mix)
    with torch.no_grad():
        pred_val_img = netI(val_image)
        pred_val_user = netU(val_user)
        pred_val_mix = netM(val_mix)

    tr_image_acc, tr_user_acc, tr_mix_acc = cal_acc(pred_tr_img.cpu(), pred_tr_user.cpu(), pred_tr_mix.cpu(), tr_y)
    val_image_acc, val_user_acc, val_mix_acc = cal_acc(pred_val_img.cpu(), pred_val_user.cpu(), pred_val_mix.cpu(), val_y)

    print("image accuracy | train: %.3f %% | test: %.3f %% " % (tr_image_acc,  val_image_acc))
    print("user accuracy  | train: %.3f %% | test: %.3f %% " % (tr_user_acc, val_user_acc))
    print("mix accuracy   | train: %.3f %% | test: %.3f %% " % (tr_mix_acc, val_mix_acc))


if __name__ == "__main__":
    main()
