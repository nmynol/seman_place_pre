import tkinter

from torch.nn.utils.rnn import *
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
#
# lst = []
#
# lst.append(torch.randn((1, 4)))
# lst.append(torch.randn((3, 4)))
# lst.append(torch.randn((5, 4)))
#
# sort_list = lst
# # sort_list = sorted(lst, key=len, reverse=True)
# list_len = list(map(len, sort_list))
# # print(sort_list)
# # print(list_len)
#
# lst = pad_sequence(sort_list, batch_first=True)
#
# print(lst.shape)
# lst_packed = pack_padded_sequence(lst, list_len[:3], batch_first=True)
#
# print(lst_packed[0].shape)
#
# lstm = nn.LSTM(4, 20, batch_first=True)
#
# out, _ = lstm(lst_packed)
# print(out[0].shape)
# pad_out, _ = pad_packed_sequence(out, batch_first=True)
#
# print(pad_out.shape)

# pred_text = torch.randn(4, 8)
# y = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0],
#                   [0, 1, 0, 0, 0, 0, 0, 0],
#                   [0, 0, 1, 0, 0, 0, 0, 0],
#                   [0, 0, 0, 1, 0, 0, 0, 0]])
#
# pred_text = (pred_text.numpy() == pred_text.numpy().max(axis=1, keepdims=1)).astype("float64")
# print(pred_text)
# pred_text = [np.argmax(item) for item in pred_text]
# print(pred_text)
# y = [np.argmax(item) for item in y]
# print(y)
# pred_text, y = np.array(pred_text), np.array(y)
# print(pred_text)
# print(y)
# per_text = pred_text == y
# print(per_text)
# text_acc = len(per_text[per_text == True]) / len(per_text) * 100
# print(text_acc)
import random

# x = np.random.randn(123, 214, 300)
# y = np.random.randn(123, 153, 300)
# x = np.pad(y, ((0, 0), (0, max(y.shape[1], x.shape[1]) - min(y.shape[1], x.shape[1])), (0, 0)),
#            'constant', constant_values=((0, 0), (0, 0), (0, 0)))
# print(x.shape)
#
# torch.nn.Embedding.from_pretrained()
