# from __future__ import print_function

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from easydict import EasyDict
import random

from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_


class BottlenetT(nn.Module):
    """
        network for text(CNN version)
        input_size: [batch_size, 153, 300]
        output_size: [batch_size, 8]
    """
    def __init__(self, args):
        super(BottlenetT, self).__init__()
        self.args = args
        self.kernel_size = list(map(int, args.kernel_sizes.split(",")))

        self.convs = nn.ModuleList([nn.Conv2d(1, args.kernel_num, (K, args.embed_dim)) for K in self.kernel_size])
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(self.kernel_size) * args.kernel_num, args.class_num)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, 153, 300)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # (batch_size, 100, 153)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # (batch_size, 100)
        x = torch.cat(x, 1)  # (batch_size, 100)
        x = self.dropout(x)  # # (batch_size, 100)
        x = self.fc1(x)  # # (batch_size, 8)
        out = self.softmax(x)
        return out


class BottlenetLSTM(nn.Module):
    """
        network for text(LSTM version)
        input_size: [batch_size, 214, 300]
        output_size: [batch_size, 8]
    """
    def __init__(self, config):
        super(BottlenetLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size=config.embed_dim, hidden_size=config.hidden_size, num_layers=config.num_layers,
                           bidirectional=True, batch_first=True)
        self.f1 = nn.Sequential(nn.Linear(214, 128),
                                nn.Dropout(),
                                nn.ReLU(),
                                nn.Linear(128, 8),
                                nn.Softmax(dim=1)
                                )

    def forward(self, x):
        x, _ = self.rnn(x)
        # x = F.dropout(x, p=0.8)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in [x]]  # (batch_size, 100)
        x = torch.cat(x, 1)
        x = self.f1(x)
        return x


class BottlenetI(nn.Module):
    """
        network for image
        input_size: [batch_size, 205]
        output_size: [batch_size, 8]
    """
    def __init__(self):
        super(BottlenetI, self).__init__()
        self.net = nn.Sequential(
            # nn.Linear(input, 256),
            # nn.Linear(256, 128),
            nn.Linear(205, 8),
            nn.Dropout(),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.net(x)
        return x


class BottlenetU(nn.Module):
    """
        network for user attribute
        input_size: [batch_size, 12]
        output_size: [batch_size, 8]
    """
    def __init__(self):
        super(BottlenetU, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(12, 8),
            nn.Dropout(),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.net(x)
        return x


class BottlenetM(nn.Module):
    """
        network for image plus user attribute
        input_size: [batch_size, 217]
        output_size: [batch_size, 8]
    """
    def __init__(self):
        super(BottlenetM, self).__init__()
        self.net = nn.Sequential(
            # nn.Linear(input, 256),
            # nn.Linear(256, 128),
            nn.Linear(217, 8),
            nn.Dropout(),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.net(x)
        return x


class BottlenetF(nn.Module):

    def __init__(self, args):
        super(BottlenetF, self).__init__()

        # text part
        self.args = args
        self.kernel_size = list(map(int, args.kernel_sizes.split(",")))

        self.convs = nn.ModuleList([nn.Conv2d(1, args.kernel_num, (K, args.embed_dim)) for K in self.kernel_size])
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(self.kernel_size) * args.kernel_num, args.class_num)
        self.softmax = nn.Softmax(dim=1)

        # image part
        self.net_image = nn.Sequential(
            # nn.Linear(input, 256),
            # nn.Linear(256, 128),
            nn.Linear(205, 8),
            nn.Dropout(),
            # nn.Softmax(dim=1)
        )

        # user part
        self.net_user = nn.Sequential(
            nn.Linear(12, 8),
            nn.Dropout(),
            # nn.Softmax(dim=1)
        )

        # fusion part
        self.net_fusion = nn.Sequential(
            nn.Linear(24, 8),
            # nn.Softmax(dim=1)
        )

    def forward(self, text, image, user):
        # text part
        text = text.unsqueeze(1)  # (batch_size, 1, 153, 300)
        text = [F.relu(conv(text)).squeeze(3) for conv in self.convs]  # (batch_size, 100, 153)
        text = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in text]  # (batch_size, 100)
        text = torch.cat(text, 1)  # (batch_size, 100)
        text = self.dropout(text)  # # (batch_size, 100)
        text = self.fc1(text)  # # (batch_size, 8)
        text = self.softmax(text)
        # image part
        image = self.net_image(image)
        # user part
        user = self.net_user(user)
        # fusion part
        fusion = torch.cat((text, image, user), dim=1)
        fusion = self.net_fusion(fusion)
        return fusion


class Origin(nn.Module):

    def __init__(self, args):
        super(Origin, self).__init__()

        # text part
        self.args = args
        self.kernel_size = list(map(int, args.kernel_sizes.split(",")))

        self.convs = nn.ModuleList([nn.Conv2d(1, args.kernel_num, (K, args.embed_dim)) for K in self.kernel_size])
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(self.kernel_size) * args.kernel_num, args.class_num)
        self.softmax = nn.Softmax(dim=1)
        self.fc = nn.Linear(317, 8)

    def forward(self, text, image, user):
        # text part
        text = text.unsqueeze(1)  # (batch_size, 1, seq_len, embed_dim)
        text = [F.relu(conv(text)).squeeze(3) for conv in self.convs]  # (batch_size, 100, seq_len)
        text = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in text]  # (batch_size, 100)
        text = torch.cat(text, 1)  # (batch_size, 100)

        out = self.fc(torch.cat((text, image, user), 1))
        return out


class Cross_Fusion(nn.Module):

    def __init__(self, args, rank):
        super(Cross_Fusion, self).__init__()

        # text part
        self.args = args
        self.kernel_size = list(map(int, args.kernel_sizes.split(",")))

        self.convs = nn.ModuleList([nn.Conv2d(1, args.kernel_num, (K, args.embed_dim)) for K in self.kernel_size])
        self.dropout = nn.Dropout(args.dropout)
        # self.fc1 = nn.Linear(len(self.kernel_size) * args.kernel_num, args.class_num)
        # self.softmax = nn.Softmax(dim=1)
        self.fc = nn.Linear(422, 8)

        self.rank = rank
        self.user_hidden = 12
        self.image_hidden = 205
        self.output_dim = 8
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

    def forward(self, user, image, text):
        """
            text: [batch_size, seq_len, embed_dim(300)]
            image: [batch_size, 205]
            user: [batch_size, 12]
        """
        # print(text)
        # text part
        text = text.unsqueeze(1)  # (batch_size, 1, seq_len, embed_dim)
        text = F.relu(self.convs[0](text)).squeeze(3)  # (batch_size, 100, seq_len)
        # text = torch.cat((text, image.unsqueeze(-1)), -1)
        text = F.max_pool1d(text, text.size(2)).squeeze(2)  # (batch_size, 100)
        # print(text)
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

        print(self.text_factor)
        print(Parameter(torch.Tensor(4, 205, 300)))
        # print(self.text_factor)
        # print(self.text_factor)

        # output = torch.sum(fusion_zy, dim=0).squeeze()
        # use linear transformation instead of simple summation, more flexibility
        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias

        output = output.view(-1, self.output_dim)
        # output = F.softmax(output, dim=1)
        return output

        # concat = torch.cat((text, image, user), 1)
        # out = self.fc(concat)
        # out = self.dropout(out)
        # return out


class BottlenetFLSTM(nn.Module):
    """
        network for image plus user attribute
        input_size: [batch_size, 24]
        output_size: [batch_size, 8]
    """
    def __init__(self):
        super(BottlenetFLSTM, self).__init__()

        # text part
        self.rnn = nn.LSTM(input_size=300, hidden_size=300, num_layers=2,
                           bidirectional=True, batch_first=True)
        self.f1 = nn.Sequential(nn.Linear(214, 128),
                                nn.Dropout(),
                                nn.ReLU(),
                                nn.Linear(128, 8),
                                nn.Softmax(dim=1)
                                )

        # image part
        self.net_image = nn.Sequential(
            # nn.Linear(input, 256),
            # nn.Linear(256, 128),
            nn.Linear(205, 8),
            nn.Dropout(),
            nn.Softmax(dim=1)
        )

        # user part
        self.net_user = nn.Sequential(
            nn.Linear(12, 8),
            nn.Dropout(),
            nn.Softmax(dim=1)
        )

        # fusion part
        self.net_fusion = nn.Sequential(
            nn.Linear(24, 8),
            nn.Softmax(dim=1)
        )

    def forward(self, text, image, user):
        # text part
        text, _ = self.rnn(text)
        print(text.shape)
        # x = F.dropout(x, p=0.8)
        text = text.permute(0, 2, 1)
        text = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in [text]]  # (batch_size, 100)
        text = torch.cat(text, 1)
        print(text.shape)
        text = self.f1(text)
        # image part
        image = self.net_image(image)
        # user part
        user = self.net_user(user)
        # fusion part
        fusion = torch.cat((text, image, user), dim=1)
        fusion = self.net_fusion(fusion)
        return fusion


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


class TextSubNet(nn.Module):
    '''
    The LSTM-based subnetwork that is used in LMF for text
    '''

    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(TextSubNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, sequence_len, in_size)
        '''
        _, final_states = self.rnn(x)
        h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        return y_1


class LMF(nn.Module):
    '''
    Low-rank Multimodal Fusion
    '''

    def __init__(self, input_dims, hidden_dims, text_out, dropouts, output_dim, rank, use_softmax=False):
        '''
        Args:
            input_dims - a length-3 tuple, contains (audio_dim, video_dim, text_dim)
            hidden_dims - another length-3 tuple, hidden dims of the sub-networks
            text_out - int, specifying the resulting dimensions of the text subnetwork
            dropouts - a length-4 tuple, contains (audio_dropout, video_dropout, text_dropout, post_fusion_dropout)
            output_dim - int, specifying the size of output
            rank - int, specifying the size of rank in LMF
        Output:
            (return value in forward) a scalar value between -3 and 3
        '''
        super(LMF, self).__init__()

        # dimensions are specified in the order of audio, video and text
        self.audio_in = input_dims[0]
        self.video_in = input_dims[1]
        self.text_in = input_dims[2]

        self.audio_hidden = hidden_dims[0]
        self.video_hidden = hidden_dims[1]
        self.text_hidden = hidden_dims[2]
        self.text_out= text_out
        self.output_dim = output_dim
        self.rank = rank
        self.use_softmax = use_softmax

        self.audio_prob = dropouts[0]
        self.video_prob = dropouts[1]
        self.text_prob = dropouts[2]
        self.post_fusion_prob = dropouts[3]

        # define the pre-fusion subnetworks
        self.audio_subnet = SubNet(self.audio_in, self.audio_hidden, self.audio_prob)
        self.video_subnet = SubNet(self.video_in, self.video_hidden, self.video_prob)
        self.text_subnet = TextSubNet(self.text_in, self.text_hidden, self.text_out, dropout=self.text_prob)

        # define the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        # self.post_fusion_layer_1 = nn.Linear((self.text_out + 1) * (self.video_hidden + 1) * (self.audio_hidden + 1), self.post_fusion_dim)
        self.audio_factor = Parameter(torch.Tensor(self.rank, self.audio_hidden + 1, self.output_dim))
        self.video_factor = Parameter(torch.Tensor(self.rank, self.video_hidden + 1, self.output_dim))
        self.text_factor = Parameter(torch.Tensor(self.rank, self.text_out + 1, self.output_dim))
        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = Parameter(torch.Tensor(1, self.output_dim))

        # init teh factors
        xavier_normal_(self.audio_factor)
        xavier_normal_(self.video_factor)
        xavier_normal_(self.text_factor)
        xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self, audio_x, video_x, text_x):
        '''
        Args:
            audio_x: tensor of shape (batch_size, audio_in)
            video_x: tensor of shape (batch_size, video_in)
            text_x: tensor of shape (batch_size, sequence_len, text_in)
        '''
        audio_h = self.audio_subnet(audio_x)
        video_h = self.video_subnet(video_x)
        text_h = self.text_subnet(text_x)
        batch_size = audio_h.data.shape[0]

        # next we perform low-rank multimodal fusion
        # here is a more efficient implementation than the one the paper describes
        # basically swapping the order of summation and elementwise product
        if audio_h.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor

        _audio_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), audio_h), dim=1)
        _video_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), video_h), dim=1)
        _text_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), text_h), dim=1)

        fusion_audio = torch.matmul(_audio_h, self.audio_factor)
        fusion_video = torch.matmul(_video_h, self.video_factor)
        fusion_text = torch.matmul(_text_h, self.text_factor)
        fusion_zy = fusion_audio * fusion_video * fusion_text

        # output = torch.sum(fusion_zy, dim=0).squeeze()
        # use linear transformation instead of simple summation, more flexibility
        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)
        if self.use_softmax:
            output = F.softmax(output, dim=1)
        return output


if __name__ == '__main__':
    # seed_num = 2333
    # torch.manual_seed(seed_num)
    # random.seed(seed_num)

    parser = argparse.ArgumentParser(description='PyTorch for image-user CNN')
    parser.add_argument('--config', default='../config.yaml')
    parser.add_argument('--resume', default='', type=str, help='path to checkpoint')  # 增加属性
    args = parser.parse_args()

    with open(args.config) as f:
        config = EasyDict(yaml.load(f))

    net = BottlenetFLSTM()
    # net = Origin(config)
    # net = LMF((12, 205, 300), (8, 32, 32), 32, (0.5, 0.5, 0, 0.5), 8, 4)

    text = torch.randn((4, 214, 300))
    image = torch.randn((4, 4096))
    user = torch.randn((4, 12))

    # print(text)
    out = net(text, image, user)
    # print(out)

    # net = BottlenetLSTM(config)
    #
    # text = torch.randn((4, 214, 300))
    # image = torch.randn((4, 205))
    # user = torch.randn((4, 12))
    # # x = net(text, image, user)
    # x = net(text)
    # print(x.shape)

    # batch_size = 147
    # seq_length = 153
    # embed_size = 200
    # hidden_size = 300
    # num_layers = 2
    #
    #
    # lstm = nn.LSTM(input_size=embed_size,  # 输入数据的特征数是4
    #                hidden_size=hidden_size,  # 输出的特征数（hidden_size）是10
    #                num_layers=num_layers,
    #                batch_first=True,
    #                bidirectional=True)  # 使用batch_first数据维度表达方式，即(batch_size, 序列长度， 特征数目)
    #
    # x = torch.randn(batch_size, seq_length, embed_size)
    # h0 = torch.randn(2 * num_layers, batch_size, hidden_size)
    # c0 = torch.randn(2 * num_layers, batch_size, hidden_size)
    # out, (h_out, c_out) = lstm(x, (h0, c0))
    # print(out.shape)
    # print(h_out.shape)
    # print(c_out.shape)
    #
    # net = BottlenetLSTM()
    # x = torch.randn(batch_size, seq_length, embed_size)
    # y = net(x)
    # print(y.shape)
