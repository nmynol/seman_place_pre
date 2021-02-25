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


def load_text(x_text, name):
    text_list = []
    for x in x_text:
        x_u = str(x)
        xlist = generate_list(x_u)
        text_list.append(xlist)
    max_document_length = max([len(text.split(" ")) for text in text_list])
    print("max_document_length in " + name + " set:")
    print(max_document_length)

    print("using word2vec to embed words in " + name + " set...")
    model = gensim.models.KeyedVectors.load_word2vec_format('./alldata/vectors100.txt', binary=False)
    print("embedding completed.")
    # print(text_list)
    all_vectors = []
    embeddingUnknown = [0 for i in range(config.embedding_dim)]

    if not os.path.exists('./alldata/x_text_' + name + '.npy'):
        print("constructing x_text_" + name + "....")
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
        x_text = np.array(all_vectors)
        print("construction completed.")
        print("saving x_text_ " + name + "...")
        np.save("./alldata/x_text_" + name + ".npy", x_text)
        print("x_text saved.")
    else:
        print("loading x_text_" + name + ".....")
        x_text = np.load("./alldata/x_text_" + name + ".npy")
        print("x_text_" + name + " loaded.")
    return x_text


class ImageFolder(data.Dataset):
    def __init__(self, x_text, y):
        self.x_text = x_text
        self.y = y
        self.x_text = torch.Tensor(self.x_text.astype("float64"))
        self.y = torch.Tensor(np.array(self.y, dtype="float64"))

    def __getitem__(self, index):
        return self.x_text[index], self.y[index]

    def __len__(self):
        return len(self.x_text)


def data_prepare(config):
    text, _, _, y = data_helpers.load_data_and_labels(config.train.home_data, config.train.work_data,
                                                            config.train.school_data,
                                                            config.train.restaurant_data,
                                                            config.train.shopping_data,
                                                            config.train.cinema_data,
                                                            config.train.sports_data,
                                                            config.train.travel_data)

    text = load_text(text, "train_100")
    permutation = np.random.permutation(y.shape[0])
    shuffled_text = text[permutation, :, :]
    shuffled_y = y[permutation, :]

    return shuffled_text[3000:shuffled_text.shape[0], :, :], shuffled_text[0:3000, :, :], \
           shuffled_y[3000:shuffled_y.shape[0], :], shuffled_y[0:3000, :]


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


def load_state(path, netT, optimizerT):
    if not os.path.isfile(path):
        print("=> no checkpoint found at '{}'".format(path))
    else:
        print("=> loading checkpoint '{}'".format(path))

        checkpoint = torch.load(path)
        netT.load_state_dict(checkpoint['state_dictT'])
        optimizerT.load_state_dict(checkpoint['optimizerT'])
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

    netT = BottlenetT(config)
    criterion = nn.CrossEntropyLoss()
    netT = netT.to(device)
    criterion = criterion.to(device)

    optimizerT = optim.Adam(netT.parameters(), lr=config.lr_text)

    last_epoch = 0
    if args.resume:
        last_epoch = load_state(args.resume, netT, optimizerT)

    tr_text_emb, val_text_emb, tr_y, val_y = data_prepare(config)

    train_dataset = ImageFolder(tr_text_emb, tr_y)
    train_dataloader = data.DataLoader(train_dataset, batch_size=config.batch_size,
                                       shuffle=True, pin_memory=True, num_workers=int(config.workers))

    for epoch in range(last_epoch, config.epoch - 1):
        for iter, [text, y] in enumerate(train_dataloader):
            netT.zero_grad()
            text, y = text.to(device), y.to(device)
            pred_text = netT(text)
            err_text = criterion(pred_text, y.argmax(dim=1))
            # print("err_text: ", err_text)
            err_text.backward()
            optimizerT.step()

        if (epoch + 1) % config.val_freq == 0:
            val(tr_text_emb, val_text_emb, tr_y, val_y, netT.eval(), device)

        if (epoch + 1) % config.save_freq == 0:
            save_state({'state_dictI': netT.state_dict(),
                        'optimizerI': optimizerT.state_dict(),
                        'epoch': epoch}, config.text_save_path, epoch)


def val(tr_text_emb, val_text_emb, tr_y, val_y, netT, device):

    tr_text = torch.Tensor(np.array(tr_text_emb, dtype="float64"))
    tr_y = np.array(tr_y, dtype="float64")
    val_text = torch.Tensor(np.array(val_text_emb, dtype="float64"))
    val_y = np.array(val_y, dtype="float64")

    tr_text = tr_text.to(device)
    val_text = val_text.to(device)

    with torch.no_grad():
        pred_tr_text = netT(tr_text)
        pred_val_text = netT(val_text)

    tr_text_acc = cal_acc(pred_tr_text.cpu(), tr_y)
    val_text_acc = cal_acc(pred_val_text.cpu(), val_y)

    print("text accuracy   | train: %.3f %% | test: %.3f %% " % (tr_text_acc, val_text_acc))


if __name__ == "__main__":
    main()
