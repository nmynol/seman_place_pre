# @Author : bamtercelboo
# @Datetime : 2018/07/19 22:35
# @File : model_CNN.py
# @Last Modify Time : 2018/07/19 22:35
# @Contact : bamtercelboo@{gmail.com, 163.com}
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_

"""
    Neural Network: CNN
"""


class SubNet(nn.Module):
    '''
    The subnetwork that is used in LMF for video and audio in the pre-fusion stage
    '''

    def __init__(self, in_size, hidden_size, dropout):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(SubNet, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        normed = self.norm(x)
        dropped = self.drop(normed)
        y_1 = F.relu(self.linear_1(dropped))
        y_2 = F.relu(self.linear_2(y_1))
        y_3 = F.relu(self.linear_3(y_2))

        return y_3


class CNN_Text(nn.Module):

    def __init__(self, args):
        super(CNN_Text, self).__init__()
        self.args = args

        # V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes

        # if args.max_norm is not None:
        #     print("max_norm = {} ".format(args.max_norm))
        #     self.embed = nn.Embedding(V, D, max_norm=5, scale_grad_by_freq=True, padding_idx=args.paddingId)
        # else:
        #     print("max_norm = {} ".format(args.max_norm))
        #     self.embed = nn.Embedding(V, D, scale_grad_by_freq=True, padding_idx=args.paddingId)
        # if args.word_Embedding:
        #     self.embed.weight.data.copy_(args.pretrained_weight)
        #     # fixed the word embedding
        #     self.embed.weight.requires_grad = True
        # print("dddd {} ".format(self.embed.weight.data.size()))

        if args.wide_conv is True:
            print("using wide convolution")
            self.convs1 = nn.ModuleList([nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(K, D), stride=(1, 1),
                                                   padding=(K // 2, 0), dilation=1, bias=False) for K in Ks])
        else:
            print("using narrow convolution")
            self.convs1 = nn.ModuleList([nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(K, D),
                                                   bias=True) for K in Ks])
        print(self.convs1)

        if args.init_weight:
            print("Initing W .......")
            for conv in self.convs1:
                init.xavier_normal_(conv.weight.data, gain=np.sqrt(args.init_weight_value))
                fan_in, fan_out = CNN_Text.calculate_fan_in_and_fan_out(conv.weight.data)
                print(" in {} out {} ".format(fan_in, fan_out))
                std = np.sqrt(args.init_weight_value) * np.sqrt(2.0 / (fan_in + fan_out))
        # for cnn cuda
        # if self.args.cuda is True:
        #     for conv in self.convs1:
        #         conv = conv.cuda()

        self.dropout = nn.Dropout(args.dropout)
        self.dropout_embed = nn.Dropout(args.dropout_embed)
        in_fea = len(Ks) * Co + 12 + 205
        self.fc = nn.Linear(in_features=in_fea, out_features=C, bias=True)
        # whether to use batch normalizations
        if args.batch_normalizations is True:
            print("using batch_normalizations in the model......")
            self.convs1_bn = nn.BatchNorm2d(num_features=Co, momentum=args.bath_norm_momentum,
                                            affine=args.batch_norm_affine)
            self.fc1_bn = nn.BatchNorm1d(num_features=in_fea // 2, momentum=args.bath_norm_momentum,
                                         affine=args.batch_norm_affine)
            self.fc2_bn = nn.BatchNorm1d(num_features=C, momentum=args.bath_norm_momentum,
                                         affine=args.batch_norm_affine)

    def calculate_fan_in_and_fan_out(tensor):
        dimensions = tensor.ndimension()
        if dimensions < 2:
            raise ValueError("Fan in and fan out can not be computed for tensor with less than 2 dimensions")

        if dimensions == 2:  # Linear
            fan_in = tensor.size(1)
            fan_out = tensor.size(0)
        else:
            num_input_fmaps = tensor.size(1)
            num_output_fmaps = tensor.size(0)
            receptive_field_size = 1
            if tensor.dim() > 2:
                receptive_field_size = tensor[0][0].numel()
            fan_in = num_input_fmaps * receptive_field_size
            fan_out = num_output_fmaps * receptive_field_size

        return fan_in, fan_out

    def forward(self, user, image, text):
        # text = self.embed(text)  # (N,W,D)
        # text = self.dropout_embed(text)
        text = text.unsqueeze(1)  # (N,Ci,W,D)
        if self.args.batch_normalizations is True:
            text = [self.convs1_bn(F.tanh(conv(text))).squeeze(3) for conv in self.convs1]  # [(N,Co,W), ...]*len(Ks)
            text = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in text]  # [(N,Co), ...]*len(Ks)
        else:
            text = [F.relu(conv(text)).squeeze(3) for conv in self.convs1]  # [(N,Co,W), ...]*len(Ks)
            text = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in text]  # [(N,Co), ...]*len(Ks)
        text = torch.cat(text, 1)
        text = self.dropout(text)  # (N,len(Ks)*Co)

        # print(text.shape)
        cat = torch.cat((user, image, text), 1)
        if self.args.batch_normalizations is True:
            cat = self.fc1_bn(self.fc1(cat))
            logit = self.fc2_bn(self.fc2(F.tanh(cat)))
        else:
            logit = self.fc(cat)
        return logit


class TUI(nn.Module):

    def __init__(self, args, rank):
        super(TUI, self).__init__()
        self.args = args
        self.rank = rank
        self.user_hidden = 12
        self.image_hidden = 205
        self.output_dim = 8

        # V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes

        # if args.max_norm is not None:
        #     print("max_norm = {} ".format(args.max_norm))
        #     self.embed = nn.Embedding(V, D, max_norm=5, scale_grad_by_freq=True, padding_idx=args.paddingId)
        # else:
        #     print("max_norm = {} ".format(args.max_norm))
        #     self.embed = nn.Embedding(V, D, scale_grad_by_freq=True, padding_idx=args.paddingId)
        # if args.word_Embedding:
        #     self.embed.weight.data.copy_(args.pretrained_weight)
        #     # fixed the word embedding
        #     self.embed.weight.requires_grad = True
        # print("dddd {} ".format(self.embed.weight.data.size()))

        if args.wide_conv is True:
            print("using wide convolution")
            self.convs1 = nn.ModuleList([nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(K, D), stride=(1, 1),
                                                   padding=(K // 2, 0), dilation=1, bias=False) for K in Ks])
        else:
            print("using narrow convolution")
            self.convs1 = nn.ModuleList([nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(K, D),
                                                   bias=True) for K in Ks])
        print(self.convs1)

        if args.init_weight:
            print("Initing W .......")
            for conv in self.convs1:
                init.xavier_normal_(conv.weight.data, gain=np.sqrt(args.init_weight_value))
                fan_in, fan_out = CNN_Text.calculate_fan_in_and_fan_out(conv.weight.data)
                print(" in {} out {} ".format(fan_in, fan_out))
                std = np.sqrt(args.init_weight_value) * np.sqrt(2.0 / (fan_in + fan_out))
        # for cnn cuda
        # if self.args.cuda is True:
        #     for conv in self.convs1:
        #         conv = conv.cuda()

        self.dropout = nn.Dropout(args.dropout)
        self.dropout_embed = nn.Dropout(args.dropout_embed)
        in_fea = len(Ks) * Co + 12 + 205
        self.fc = nn.Linear(in_features=in_fea, out_features=C, bias=True)
        # whether to use batch normalizations
        if args.batch_normalizations is True:
            print("using batch_normalizations in the model......")
            self.convs1_bn = nn.BatchNorm2d(num_features=Co, momentum=args.bath_norm_momentum,
                                            affine=args.batch_norm_affine)
            self.fc1_bn = nn.BatchNorm1d(num_features=in_fea // 2, momentum=args.bath_norm_momentum,
                                         affine=args.batch_norm_affine)
            self.fc2_bn = nn.BatchNorm1d(num_features=C, momentum=args.bath_norm_momentum,
                                         affine=args.batch_norm_affine)

        self.user_factor = Parameter(torch.Tensor(self.rank, self.user_hidden + 1, self.output_dim))
        self.image_factor = Parameter(torch.Tensor(self.rank, self.image_hidden + 1, self.output_dim))
        self.text_factor = Parameter(torch.Tensor(self.rank, args.kernel_num + 1, self.output_dim))
        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = Parameter(torch.Tensor(1, self.output_dim))

        xavier_normal_(self.user_factor)
        xavier_normal_(self.image_factor)
        xavier_normal_(self.text_factor)
        xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def calculate_fan_in_and_fan_out(tensor):
        dimensions = tensor.ndimension()
        if dimensions < 2:
            raise ValueError("Fan in and fan out can not be computed for tensor with less than 2 dimensions")

        if dimensions == 2:  # Linear
            fan_in = tensor.size(1)
            fan_out = tensor.size(0)
        else:
            num_input_fmaps = tensor.size(1)
            num_output_fmaps = tensor.size(0)
            receptive_field_size = 1
            if tensor.dim() > 2:
                receptive_field_size = tensor[0][0].numel()
            fan_in = num_input_fmaps * receptive_field_size
            fan_out = num_output_fmaps * receptive_field_size

        return fan_in, fan_out

    def forward(self, user, image, text):
        # text = self.embed(text)  # (N,W,D)
        # text = self.dropout_embed(text)
        text = text.unsqueeze(1)  # (N,Ci,W,D)
        if self.args.batch_normalizations is True:
            text = [self.convs1_bn(F.tanh(conv(text))).squeeze(3) for conv in self.convs1]  # [(N,Co,W), ...]*len(Ks)
            text = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in text]  # [(N,Co), ...]*len(Ks)
        else:
            text = [F.relu(conv(text)).squeeze(3) for conv in self.convs1]  # [(N,Co,W), ...]*len(Ks)
            text = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in text]  # [(N,Co), ...]*len(Ks)
        text = torch.cat(text, 1)
        text = self.dropout(text)  # (N,len(Ks)*Co)

        '''
            position of dropout
        '''

        # # origin part
        # cat = torch.cat((user, image, text), 1)
        #
        # if self.args.batch_normalizations is True:
        #     cat = self.fc1_bn(self.fc1(cat))
        #     logit = self.fc2_bn(self.fc2(F.tanh(cat)))
        # else:
        #     logit = self.fc(cat)

        # lmf part
        batch_size = text.data.shape[0]
        if text.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor

        _text = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), text), dim=1)
        _image = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), image), dim=1)
        _user = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), user), dim=1)

        fusion_text = torch.matmul(_text, self.text_factor)
        fusion_image = torch.matmul(_image, self.image_factor)
        fusion_user = torch.matmul(_user, self.user_factor)
        fusion_zy = fusion_text * fusion_image * fusion_user

        # output = torch.sum(fusion_zy, dim=0).squeeze()
        # use linear transformation instead of simple summation, more flexibility
        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias

        logit = output.view(-1, self.output_dim)

        return logit


class TUI_img_user(nn.Module):

    def __init__(self, args, rank):
        super(TUI_img_user, self).__init__()
        self.args = args
        self.rank = rank
        self.user_hidden = 8
        self.image_hidden = 128
        self.output_dim = 8

        # V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes

        self.user_subnet = SubNet(12, self.user_hidden, self.args.dropout)
        self.image_subnet = SubNet(205, self.image_hidden, self.args.dropout)

        # if args.max_norm is not None:
        #     print("max_norm = {} ".format(args.max_norm))
        #     self.embed = nn.Embedding(V, D, max_norm=5, scale_grad_by_freq=True, padding_idx=args.paddingId)
        # else:
        #     print("max_norm = {} ".format(args.max_norm))
        #     self.embed = nn.Embedding(V, D, scale_grad_by_freq=True, padding_idx=args.paddingId)
        # if args.word_Embedding:
        #     self.embed.weight.data.copy_(args.pretrained_weight)
        #     # fixed the word embedding
        #     self.embed.weight.requires_grad = True
        # print("dddd {} ".format(self.embed.weight.data.size()))

        if args.wide_conv is True:
            print("using wide convolution")
            self.convs1 = nn.ModuleList([nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(K, D), stride=(1, 1),
                                                   padding=(K // 2, 0), dilation=1, bias=False) for K in Ks])
        else:
            print("using narrow convolution")
            self.convs1 = nn.ModuleList([nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(K, D),
                                                   bias=True) for K in Ks])
        print(self.convs1)

        if args.init_weight:
            print("Initing W .......")
            for conv in self.convs1:
                init.xavier_normal_(conv.weight.data, gain=np.sqrt(args.init_weight_value))
                fan_in, fan_out = CNN_Text.calculate_fan_in_and_fan_out(conv.weight.data)
                print(" in {} out {} ".format(fan_in, fan_out))
                std = np.sqrt(args.init_weight_value) * np.sqrt(2.0 / (fan_in + fan_out))
        # for cnn cuda
        # if self.args.cuda is True:
        #     for conv in self.convs1:
        #         conv = conv.cuda()

        self.dropout = nn.Dropout(args.dropout)
        self.dropout_embed = nn.Dropout(args.dropout_embed)
        in_fea = len(Ks) * Co + 12 + 205
        self.fc = nn.Linear(in_features=in_fea, out_features=C, bias=True)
        # whether to use batch normalizations
        if args.batch_normalizations is True:
            print("using batch_normalizations in the model......")
            self.convs1_bn = nn.BatchNorm2d(num_features=Co, momentum=args.bath_norm_momentum,
                                            affine=args.batch_norm_affine)
            self.fc1_bn = nn.BatchNorm1d(num_features=in_fea // 2, momentum=args.bath_norm_momentum,
                                         affine=args.batch_norm_affine)
            self.fc2_bn = nn.BatchNorm1d(num_features=C, momentum=args.bath_norm_momentum,
                                         affine=args.batch_norm_affine)

        self.user_factor = Parameter(torch.Tensor(self.rank, self.user_hidden + 1, self.output_dim))
        self.image_factor = Parameter(torch.Tensor(self.rank, self.image_hidden + 1, self.output_dim))
        self.text_factor = Parameter(torch.Tensor(self.rank, args.kernel_num + 1, self.output_dim))
        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = Parameter(torch.Tensor(1, self.output_dim))

        xavier_normal_(self.fc.weight)
        xavier_normal_(self.user_factor)
        xavier_normal_(self.image_factor)
        xavier_normal_(self.text_factor)
        xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def calculate_fan_in_and_fan_out(tensor):
        dimensions = tensor.ndimension()
        if dimensions < 2:
            raise ValueError("Fan in and fan out can not be computed for tensor with less than 2 dimensions")

        if dimensions == 2:  # Linear
            fan_in = tensor.size(1)
            fan_out = tensor.size(0)
        else:
            num_input_fmaps = tensor.size(1)
            num_output_fmaps = tensor.size(0)
            receptive_field_size = 1
            if tensor.dim() > 2:
                receptive_field_size = tensor[0][0].numel()
            fan_in = num_input_fmaps * receptive_field_size
            fan_out = num_output_fmaps * receptive_field_size

        return fan_in, fan_out

    def forward(self, user, image, text):
        # text = self.embed(text)  # (N,W,D)
        # text = self.dropout_embed(text)
        text = text.unsqueeze(1)  # (N,Ci,W,D)
        if self.args.batch_normalizations is True:
            text = [self.convs1_bn(F.tanh(conv(text))).squeeze(3) for conv in self.convs1]  # [(N,Co,W), ...]*len(Ks)
            text = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in text]  # [(N,Co), ...]*len(Ks)
        else:
            text = [F.relu(conv(text)).squeeze(3) for conv in self.convs1]  # [(N,Co,W), ...]*len(Ks)
            text = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in text]  # [(N,Co), ...]*len(Ks)
        text = torch.cat(text, 1)
        # text = self.dropout(text)  # (N,len(Ks)*Co)

        # origin part

        # user = self.user_subnet(user)
        # image = self.image_subnet(image)
        cat = torch.cat((user, image, text), 1)
        cat = self.dropout(cat)

        if self.args.batch_normalizations is True:
            cat = self.fc1_bn(self.fc1(cat))
            logit = self.fc2_bn(self.fc2(F.tanh(cat)))
        else:
            logit = self.fc(cat)

        # # lmf part
        #
        # user = self.user_subnet(user)
        # image = self.image_subnet(image)
        #
        # batch_size = text.data.shape[0]
        # if text.is_cuda:
        #     DTYPE = torch.cuda.FloatTensor
        # else:
        #     DTYPE = torch.FloatTensor
        #
        # _text = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), text), dim=1)
        # _image = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), image), dim=1)
        # _user = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), user), dim=1)
        #
        # fusion_text = torch.matmul(_text, self.text_factor)
        # fusion_image = torch.matmul(_image, self.image_factor)
        # fusion_user = torch.matmul(_user, self.user_factor)
        # fusion_zy = fusion_text * fusion_image * fusion_user
        #
        # # output = torch.sum(fusion_zy, dim=0).squeeze()
        # # use linear transformation instead of simple summation, more flexibility
        # output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        #
        # logit = output.view(-1, self.output_dim)

        return logit


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch for image-user CNN')
    # parser.add_argument('--embed_num', default=)
    parser.add_argument('--embed_dim', default=300)
    parser.add_argument('--class_num', default=8)
    parser.add_argument('--kernel_num', default=100)
    parser.add_argument('--kernel_sizes', default=[1])
    # parser.add_argument('--max_norm', default=)
    # parser.add_argument('--paddingId', default=)
    # parser.add_argument('--word_Embedding', default=)
    # parser.add_argument('--pretrained_weight', default=)
    parser.add_argument('--wide_conv', default=True)
    parser.add_argument('--init_weight', default=True)
    parser.add_argument('--init_weight_value', default=2.0)
    # parser.add_argument('--cuda', default=False)
    parser.add_argument('--dropout', default=0.75)
    parser.add_argument('--dropout_embed', default=0.75)
    parser.add_argument('--batch_normalizations', default=False)
    parser.add_argument('--batch_norm_affine', default=False)
    config = parser.parse_args()

    net = TUI_img_user(config, 4)

    user = torch.randn((4, 12))
    image = torch.randn((4, 205))
    text = torch.randn((4, 214, 300))
    # print(text)
    y = net(user, image, text)
    print(y.shape)
