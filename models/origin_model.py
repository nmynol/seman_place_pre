import gensim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_
import numpy as np


class CNN_Text(nn.Module):

    def __init__(self):
        super(CNN_Text, self).__init__()

        self.embed_dim = 300
        self.img_dim = 4096
        self.user_dim = 12
        self.class_num = 8
        self.Ci = 1
        self.kernel_num = 100
        self.kernel_sizes = [1]
        self.dropout_rate = 0.5

        self.convs1 = nn.ModuleList([nn.Conv2d(in_channels=self.Ci, out_channels=self.kernel_num,
                                               kernel_size=(K, self.embed_dim), bias=True) for K in self.kernel_sizes])

        self.dropout = nn.Dropout(self.dropout_rate)
        in_fea = len(self.kernel_sizes) * self.kernel_num + self.img_dim + self.user_dim
        self.fc = nn.Linear(in_features=in_fea, out_features=self.class_num, bias=True)
        torch.nn.init.xavier_normal_(self.fc.weight.data)

    def forward(self, text, image, user):
        text = text.unsqueeze(1)  # (N,Ci,W,D)
        text = [F.relu(conv(text)).squeeze(3) for conv in self.convs1]  # [(N,Co,W), ...]*len(Ks)
        text = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in text]  # [(N,Co), ...]*len(Ks)
        text = torch.cat(text, 1)
        fusion_feature = torch.cat((text, image, user), 1)
        fusion_feature = self.dropout(fusion_feature)  # (N,len(Ks)*Co)
        logit = self.fc(fusion_feature)
        return logit


class SubNet(nn.Module):
    def __init__(self, in_size, hidden_size, dropout):
        super(SubNet, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        normed = self.norm(x)
        dropped = self.drop(normed)
        y_1 = F.relu(self.linear_1(dropped))
        y_2 = F.relu(self.linear_2(y_1))
        y_3 = F.relu(self.linear_3(y_2))
        return y_3


class LMF(nn.Module):
    def __init__(self, input_dims, hidden_dims, text_out, dropouts, output_dim, rank):
        '''
        Args:
            input_dims - a length-3 tuple, contains (audio_dim, video_dim, text_dim)
            hidden_dims - another length-2 tuple, hidden dims of the sub-networks
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
        self.text_out = text_out
        self.output_dim = output_dim
        self.rank = rank

        self.audio_prob = dropouts[0]
        self.video_prob = dropouts[1]
        self.post_fusion_prob = dropouts[2]

        # define the pre-fusion subnetworks
        self.audio_subnet = SubNet(self.audio_in, self.audio_hidden, self.audio_prob)
        self.video_subnet = SubNet(self.video_in, self.video_hidden, self.video_prob)

        # define the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
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

        self.dropout = nn.Dropout(self.post_fusion_prob)
        in_fea = 116
        self.fc = nn.Linear(in_features=in_fea, out_features=self.output_dim, bias=True)
        torch.nn.init.xavier_normal_(self.fc.weight.data)

    def forward(self, audio_x, video_x, text_x):

        audio_h = self.audio_subnet(audio_x)
        video_h = self.video_subnet(video_x)
        text_h = text_x

        # fusion_feature = torch.cat((audio_h, video_h, text_h), 1)
        # fusion_feature = self.dropout(fusion_feature)  # (N,len(Ks)*Co)
        # output = self.fc(fusion_feature)
        batch_size = text_h.data.shape[0]
        if text_h.is_cuda:
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
        # fusion_zy = fusion_audio * fusion_text
        fusion_zy = self.post_fusion_dropout(fusion_zy)
        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)

        return output


class MAG(nn.Module):
    def __init__(self, first_dim, second_dim, third_dim, class_num, beta_shift=1.0, dropout_prob=0.5):
        super(MAG, self).__init__()

        TEXT_DIM = first_dim
        VISUAL_DIM = second_dim
        ACOUSTIC_DIM = third_dim
        CLASS_NUM = class_num
        self.DEVICE = torch.device("cuda")

        self.W_hv = nn.Linear(VISUAL_DIM + TEXT_DIM, TEXT_DIM)
        self.W_ha = nn.Linear(ACOUSTIC_DIM + TEXT_DIM, TEXT_DIM)

        self.W_v = nn.Linear(VISUAL_DIM, TEXT_DIM)
        self.W_a = nn.Linear(ACOUSTIC_DIM, TEXT_DIM)
        self.beta_shift = beta_shift

        self.LayerNorm = nn.LayerNorm(TEXT_DIM)
        self.dropout = nn.Dropout(dropout_prob)

        self.fc = nn.Linear(TEXT_DIM, CLASS_NUM)

    def forward(self, text_embedding, visual, acoustic):
        eps = 1e-6
        weight_v = F.relu(self.W_hv(torch.cat((visual, text_embedding), dim=-1)))
        weight_a = F.relu(self.W_ha(torch.cat((acoustic, text_embedding), dim=-1)))

        h_m = weight_v * self.W_v(visual) + weight_a * self.W_a(acoustic)

        em_norm = text_embedding.norm(2, dim=-1)
        hm_norm = h_m.norm(2, dim=-1)

        hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True).to(self.DEVICE)
        hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)

        thresh_hold = (em_norm / (hm_norm + eps)) * self.beta_shift

        ones = torch.ones(thresh_hold.shape, requires_grad=True).to(self.DEVICE)

        alpha = torch.min(thresh_hold, ones)
        alpha = alpha.unsqueeze(dim=-1)

        acoustic_vis_embedding = alpha * h_m

        output = self.dropout(
            self.LayerNorm(acoustic_vis_embedding + text_embedding)
        )

        output = self.fc(output)

        return output


class CNN_fusion(nn.Module):

    def __init__(self, fusion_mode):
        super(CNN_fusion, self).__init__()
        self.fusion_method = fusion_mode
        # Text_CNN part
        self.embed_dim = 400
        self.img_dim = 4096
        self.user_dim = 12
        self.class_num = 8
        self.Ci = 1
        self.kernel_num = 100
        self.kernel_sizes = [1]
        self.convs1 = nn.ModuleList([nn.Conv2d(in_channels=self.Ci, out_channels=self.kernel_num,
                                               kernel_size=(K, self.embed_dim), bias=True) for K in self.kernel_sizes])
        self.rnn = nn.LSTM(input_size=300, hidden_size=50, num_layers=2,
                           bidirectional=True, batch_first=True)
        if self.fusion_method == 'simple':
            print('fusion mode: simple')
            self.dropout_rate = 0.5
            self.dropout = nn.Dropout(self.dropout_rate)
            # in_fea = len(self.kernel_sizes) * self.kernel_num + self.img_dim + self.user_dim
            in_fea = 4096 + 12 + 100
            self.fc = nn.Linear(in_features=in_fea, out_features=self.class_num, bias=True)
            torch.nn.init.xavier_normal_(self.fc.weight.data)

        elif self.fusion_method == 'LMF':
            print('fusion mode: LMF')
            self.user_hidden = 8
            self.image_hidden = 64
            self.user_prob = 0.5
            self.image_prob = 0.5
            self.post_fusion_prob = 0.5
            self.rank = 8
            self.lmf = LMF((self.user_dim, self.img_dim, self.embed_dim),
                           (self.user_hidden, self.image_hidden), len(self.kernel_sizes) * self.kernel_num,
                           (self.image_prob, self.image_prob, self.post_fusion_prob),
                           self.class_num, self.rank)

        elif self.fusion_method == 'MAG':
            print('fusion mode: MAG')
            self.dropout_rate = 0.5

            self.dropout = nn.Dropout(self.dropout_rate)
            # self.fc = nn.Linear(self.user_dim + self.kernel_num, self.class_num)
            self.mag1 = MAG(self.img_dim, self.user_dim, self.kernel_num, self.class_num)
            # self.mag2 = MAG(self.kernel_num, self.img_dim, self.user_dim, self.class_num)
            # self.mag3 = MAG(self.user_dim, self.img_dim, self.kernel_num, self.class_num)

    def forward(self, text, image, user):
        # Text_CNN part
        text = text.unsqueeze(1)  # (N,Ci,W,D)
        text = [F.relu(conv(text)).squeeze(3) for conv in self.convs1]  # [(N,Co,W), ...]*len(Ks)

        # text, _ = self.rnn(text)
        # text = text.permute(0, 2, 1)
        # text = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in [text]]  # (batch_size, 100)

        text = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in text]  # [(N,Co), ...]*len(Ks)

        text = torch.cat(text, 1)

        if self.fusion_method == 'simple':
            fusion_feature = torch.cat((text, image, user), 1)
            # fusion_feature = torch.cat((text, image), 1)
            # fusion_feature = torch.cat((image, user), 1)
            # fusion_feature = image
            fusion_feature = self.dropout(fusion_feature)  # (N,len(Ks)*Co)
            logit = self.fc(fusion_feature)

        elif self.fusion_method == 'LMF':
            logit = self.lmf(user, image, text)

        elif self.fusion_method == 'MAG':
            logit = self.mag1(image, user, text)
            # logit2 = self.mag2(text, image, user)
            # logit3 = self.mag3(user, image, text)
            # logit = self.fc(self.dropout(torch.cat((logit2, logit3), 1)))

        return logit


class CNN_fusion_emb(nn.Module):

    def __init__(self, fusion_mode):
        super(CNN_fusion_emb, self).__init__()
        self.fusion_method = fusion_mode
        # Text_CNN part
        self.embed_dim = 300
        self.img_dim = 4096
        self.user_dim = 12
        self.class_num = 8
        self.Ci = 1
        self.kernel_num = 100
        self.kernel_sizes = [1]
        self.convs1 = nn.ModuleList([nn.Conv2d(in_channels=self.Ci, out_channels=self.kernel_num,
                                               kernel_size=(K, self.embed_dim), bias=True) for K in self.kernel_sizes])
        if self.fusion_method == 'simple':
            self.dropout_rate = 0.5
            self.dropout = nn.Dropout(self.dropout_rate)
            in_fea = len(self.kernel_sizes) * self.kernel_num + self.img_dim + self.user_dim
            self.fc = nn.Linear(in_features=in_fea, out_features=self.class_num, bias=True)
            torch.nn.init.xavier_normal_(self.fc.weight.data)

        elif self.fusion_method == 'LMF':
            self.user_hidden = 8
            self.image_hidden = 8
            self.user_prob = 0.5
            self.image_prob = 0.5
            self.post_fusion_prob = 0.5
            self.rank = 4
            self.lmf = LMF((self.user_dim, self.img_dim, self.embed_dim),
                           (self.user_hidden, self.image_hidden), len(self.kernel_sizes) * self.kernel_num,
                           (self.image_prob, self.image_prob, self.post_fusion_prob),
                           self.class_num, self.rank)

    def forward(self, text, image, user):
        # Text_CNN part
        text = self.load_text(text)
        text = text.unsqueeze(1)  # (N,Ci,W,D)

        text = [F.relu(conv(text)).squeeze(3) for conv in self.convs1]  # [(N,Co,W), ...]*len(Ks)
        text = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in text]  # [(N,Co), ...]*len(Ks)
        text = torch.cat(text, 1)

        if self.fusion_method == 'simple':
            fusion_feature = torch.cat((text, image, user), 1)
            fusion_feature = self.dropout(fusion_feature)  # (N,len(Ks)*Co)
            logit = self.fc(fusion_feature)

        elif self.fusion_method == 'LMF':
            logit = self.lmf(user, image, text)

        return logit

    def generate_list(self, text_raw):
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

    def load_text(self, x_text):
        text_list = []
        for x in x_text:
            x_u = str(x)
            xlist = self.generate_list(x_u)
            text_list.append(xlist)
        max_document_length = max([len(text.split(" ")) for text in text_list])

        print("start embedding...")
        model = gensim.models.KeyedVectors.load_word2vec_format('./alldata/vectors300.txt', binary=False)
        print("embedding completed.")
        # print(text_list)
        all_vectors = []
        embeddingUnknown = [0 for i in range(self.embed_dim)]
        print("start constructing...")
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
        return x_text


if __name__ == '__main__':

    net = CNN_fusion('simple')

    user = torch.randn((58, 12))
    image = torch.randn((58, 4096))
    text = torch.randn((58, 214, 300))
    # print(text)
    y = net(text, image, user)
    print(y.shape)
