import sys
import re
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import torch.distributed as disp
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.utils.data import DataLoader

seed = 10

random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
from os.path import isfile
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
# import matplotlib.ticker as ticker




Train_file = './data/train.txt'
Test_file = './data/test.txt'
# Whole_file = './data/whole.txt'
q_vocab = './data/vocab.q.txt'
f_vocab = './data/vocab.f.txt'
PAD = "<PAD>" # padding
EOS = "<EOS>" # end of sequence
SOS = "<SOS>" # start of sequence
PAD_IDX = 0
EOS_IDX = 1
SOS_IDX = 2
BATCH_SIZE = 1
EMBED_SIZE = 1024
NUM_ENC_LAYERS = 5
NUM_DEC_LAYERS = 5
NUM_HEADS = 8 # number of heads
DK = EMBED_SIZE // NUM_HEADS # dimension of key
DV = EMBED_SIZE // NUM_HEADS # dimension of value
DROPOUT = 0.4
#src_vocab = {PAD: PAD_IDX, EOS: EOS_IDX, SOS: SOS_IDX}
#tgt_vocab = {PAD: PAD_IDX, EOS: EOS_IDX, SOS: SOS_IDX}
def normalize(x):
    #x = re.sub("[^ a-zA-Z0-9\uAC00-\uD7A3]+", " ", x)
    x = re.sub("\s+", " ", x)
    x = re.sub("^ | $", "", x)
    x = x.lower()
    return x

def tokenize(x, unit):
    #x = normalize(x)
    if unit == "char":
        x = re.sub(" ", "", x)
        return list(x)
    if unit == "word":
        return x.split(" ")
def read_vocab(filename):
    t2i = {PAD: PAD_IDX, EOS: EOS_IDX, SOS: SOS_IDX}
    with open(filename) as target:
        for line in target:
            token = line.strip().split()[0]
            if token not in t2i:
                t2i[token] = len(t2i)
    return t2i
def creat_vocab(file):
    fo = open(file)
    for line in fo:
        src, tgt = line.strip().split("\t")
        src_tokens = tokenize(src, "word")
        tgt_tokens = tokenize(tgt, "word")
        for word in src_tokens:
            if word not in src_vocab:
                src_vocab[word] = len(src_vocab)
        for word in tgt_tokens:
            if word not in tgt_vocab:
                tgt_vocab[word] = len(tgt_vocab)
    fo.close()
    return src_vocab, tgt_vocab
src_vocab = read_vocab(q_vocab)
tgt_vocab = read_vocab(f_vocab)
#src_vocab, tgt_vocab = creat_vocab(Train_file)
#src_vocab, tgt_vocab = creat_vocab(Test_file)
print(len(src_vocab))
print(len(tgt_vocab))
def showPlot(points, fig_name, extra_info):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    # loc = ticker.MultipleLocator(base=0.2)
    # ax.yaxis.set_major_locator(loc)
    plt.title(extra_info)
    plt.plot(points)
    plt.savefig("{}.png".format(fig_name))
    plt.close('all')
def load_data(file):
    data = []
    dataset = []
    fo = open(file)
    for line in fo:
        src, tgt = line.strip().split("\t")
        src_tokens = src.split()#(tokenize(src, "word")
        tgt_tokens = tgt.split()#tokenize(tgt, "word")
        dataset.append((src_tokens, tgt_tokens))
        src_seq = []
        tgt_seq = []
        for word in src_tokens:
            if word in src_vocab:
                src_seq.append(src_vocab[word])
        for word in tgt_tokens:
            if word in tgt_vocab:
                tgt_seq.append(tgt_vocab[word])
        data.append((src_seq, tgt_seq))
    #data.sort(key = lambda x: -len(x[0])) # sort by source sequence length
    fo.close()
    return data, dataset

def prepare_data(file, device):
    data, dataset = load_data(file)
    data_tensor=[]
    src_batch = []
    tgt_batch = []
    src_batch_len = 0
    tgt_batch_len = 0
    print("loading data...")
    for x, y in data:
        src = [int(i) for i in x] + [EOS_IDX]
        tgt = [int(i) for i in y] + [EOS_IDX]
        for seq in src:
            if len(src) > src_batch_len:
                src_batch_len = len(src)
        if len(tgt) > tgt_batch_len:
            tgt_batch_len = len(tgt)
        src_batch.append(src)
        tgt_batch.append(tgt)
    src_pad = [seq + [PAD_IDX] * (src_batch_len - len(seq)) for seq in src_batch]
    tgt_pad = [seq + [PAD_IDX] * (tgt_batch_len - len(seq)) for seq in tgt_batch]
    for i, j in zip(src_pad, tgt_pad):
        data_tensor.append((torch.LongTensor([i]).to(device), torch.LongTensor([j]).to(device)))
    return dataset, data_tensor
class encoder(nn.Module):
    def __init__(self, vocab_size, device):
        super().__init__()

        # architecture
        self.embed = nn.Embedding(vocab_size, EMBED_SIZE, padding_idx = PAD_IDX)
        self.pe = pos_encoder(device) # positional encoding
        self.layers = nn.ModuleList([enc_layer(device) for _ in range(NUM_ENC_LAYERS)])


    def forward(self, x, mask):
        #break_probs = []
        x = self.embed(x)
        h = x + self.pe(x.size(1))
        group_prob = 0.
        for layer in self.layers:
            h, break_prob = layer(h, mask, group_prob)
            #beak_probs.append(break_prob)
        return h
class decoder(nn.Module):
    def __init__(self, vocab_size, device):
        super().__init__()

        # architecture
        self.embed = nn.Embedding(vocab_size, EMBED_SIZE, padding_idx = PAD_IDX)
        self.pe = pos_encoder(device) # positional encoding
        self.layers = nn.ModuleList([dec_layer(device) for _ in range(NUM_DEC_LAYERS)])
        self.out = nn.Linear(EMBED_SIZE, vocab_size)
        self.softmax = nn.LogSoftmax(1)
        self.device = device

    def forward(self, enc_out, dec_in, src_mask):
        x = self.embed(dec_in)
        h = x + self.pe(x.size(1))
        mask1 = tgt_tril_mask(dec_in, self.device)
        group_prob = 0.
        for layer in self.layers:
            h, break_prob = layer(enc_out, h, src_mask, mask1, group_prob)
        h = self.out(h[:, -1])
        y = self.softmax(h)
        return y
class enc_layer(nn.Module): # encoder layer
    def __init__(self, device):
        super().__init__()

        # architecture
        self.attn = attn_mh() # self-attention
        self.ffn = ffn(2048)
        self.group_attn = GroupAttention(EMBED_SIZE, device)

    def forward(self, x, src_mask, group_prob):
        group_prob, break_prob = self.group_attn(x, x, src_mask, group_prob)
        z = self.attn(x, x, x, src_mask, group_prob)
        z = self.ffn(z)
        return z, break_prob

class dec_layer(nn.Module): # decoder layer
    def __init__(self, device):
        super().__init__()

        # architecture
        self.attn1 = attn_mh() # masked self-attention
        self.attn2 = attn_mh() # encoder-decoder attention
        self.group_attn = GroupAttention(EMBED_SIZE, device)
        self.ffn = ffn(2048)

    def forward(self, enc_out, dec_in, src_mask, mask1, group_prob):
        group_prob1, break_prob1 = self.group_attn(dec_in, dec_in, mask1, group_prob)
        z = self.attn1(dec_in, dec_in, dec_in, mask1, group_prob = None)
        group_prob2, break_prob2 = self.group_attn(enc_out, enc_out, src_mask, group_prob)
        z = self.attn2(z, enc_out, enc_out, src_mask, group_prob=None)
        z = self.ffn(z)
        return z, break_prob1
class pos_encoder(nn.Module): # positional encoding
    def __init__(self, device, maxlen = 200):
        super().__init__()
        self.pe = torch.Tensor(maxlen, EMBED_SIZE).to(device)
        pos = torch.arange(0, maxlen, 1.).unsqueeze(1)
        k = torch.exp(np.log(10000) * -torch.arange(0, EMBED_SIZE, 2.) / EMBED_SIZE)
        self.pe[:, 0::2] = torch.sin(pos * k)
        self.pe[:, 1::2] = torch.cos(pos * k)

    def forward(self, n):
        return self.pe[:n]


class GroupAttention(nn.Module):
    def __init__(self, d_model, device, dropout=0.4):
        super(GroupAttention, self).__init__()
        self.d_model = 1024
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_query = nn.Linear(d_model, d_model)
        # self.linear_output = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, q, k, pad_mask,  prior):
        batch_size, seq_len = q.size()[:2]
        #print(context.shape)
        q = self.norm(q)
        k = self.norm(k)
        a = torch.diag(torch.ones(seq_len - 1), 1).to(self.device)
        b = torch.diag(torch.ones(seq_len - 1), -1).to(self.device)
        c = torch.diag(torch.ones(seq_len - 1, dtype=torch.int32), -1).to(self.device)
        tri_matrix = torch.triu(torch.ones([seq_len, seq_len], dtype=torch.float32), 0).to(self.device)
        mask = pad_mask.squeeze(1) & (a+c)

        key = self.linear_key(k)
        query = self.linear_query(q)

        scores = torch.matmul(query, key.transpose(-2, -1)) / self.d_model
        mask = mask.to(self.device)
        scores = scores.masked_fill(mask == 0, -1e9)
        neibor_attn = F.softmax(scores, dim=-1)
        neibor_attn = torch.sqrt(neibor_attn * neibor_attn.transpose(-2, -1) + 1e-9)
        neibor_attn = prior + (1. - prior) * neibor_attn
        #print(neibor_attn.shape)
        #print(tri_matrix.shape)
        t = torch.log(neibor_attn + 1e-9).masked_fill(a == 0, 0).matmul(tri_matrix)
        g_attn = tri_matrix.matmul(t).exp().masked_fill((tri_matrix.int() - b) == 0, 0)
        g_attn = g_attn + g_attn.transpose(-2, -1) + neibor_attn.masked_fill(b == 0, 1e-9)
        return g_attn, neibor_attn

class attn_mh(nn.Module): # multi-head attention
    def __init__(self):
        super().__init__()

        # architecture
        self.Wq = nn.Linear(EMBED_SIZE, NUM_HEADS * DK) # query
        self.Wk = nn.Linear(EMBED_SIZE, NUM_HEADS * DK) # key for attention distribution
        self.Wv = nn.Linear(EMBED_SIZE, NUM_HEADS * DV) # value for context representation
        self.Wo = nn.Linear(EMBED_SIZE, NUM_HEADS * DV)
        self.dropout = nn.Dropout(DROPOUT)
        self.norm = nn.LayerNorm(EMBED_SIZE)

    def attn_sdp(self, q, k, v, mask, group_prob=None): # scaled dot-product attention
        c = np.sqrt(DK) # scale factor
        a = torch.matmul(q, k.transpose(2, 3)) / c # compatibility function
        a = a.masked_fill(mask==0, -1e9) # masking in log space
        if group_prob is not None:
            a = F.softmax(a, 3)
            a = a*group_prob.unsqueeze(1)
        else:
            a = F.softmax(a, 3)
        a = torch.matmul(a, v)
        return a # attention weights

    def forward(self, q, k, v, mask, group_prob):
        x = q # identity
        q = self.Wq(q).view(BATCH_SIZE, -1, NUM_HEADS, DK).transpose(1, 2)
        k = self.Wk(k).view(BATCH_SIZE, -1, NUM_HEADS, DK).transpose(1, 2)
        v = self.Wv(v).view(BATCH_SIZE, -1, NUM_HEADS, DV).transpose(1, 2)
        z = self.attn_sdp(q, k, v, mask, group_prob)
        z = z.transpose(1, 2).contiguous().view(BATCH_SIZE, -1, NUM_HEADS * DV)
        z = self.Wo(z)
        z = self.norm(x + self.dropout(z)) # residual connection and dropout
        return z
class ffn(nn.Module): # position-wise feed-forward networks
    def __init__(self, d):
        super().__init__()

        # architecture
        self.layers = nn.Sequential(
            nn.Linear(EMBED_SIZE, d),
            nn.ReLU(),
            nn.Linear(d, EMBED_SIZE),
        )
        self.dropout = nn.Dropout(DROPOUT)
        self.norm = nn.LayerNorm(EMBED_SIZE)

    def forward(self, x):
        z = self.layers(x)
        z = self.norm(x + self.dropout(z)) # residual connection and dropout
        return z
#def Tensor(*args):
#    x = torch.Tensor(*args)
#    return x.cuda() if CUDA else x

#def LongTensor(*args):
#    x = torch.LongTensor(*args)
#    return x.cuda() if CUDA else x

def scalar(x):
    return x.view(-1).data.tolist()[0]



#def mask_pad(x): # mask out padded positions #
#    return x.data.eq(PAD_IDX).view(BATCH_SIZE, 1, 1, -1)

#def mask_pad(x):
#    return (x != 0).type(torch.int32).view(BATCH_SIZE, 1, 1, -1)

#def mask_triu(x, device): # mask out subsequent positions
#    y = torch.Tensor(np.triu(np.ones([x.size(2), x.size(2)]), 1)).byte().to(device)
#    return torch.gt(x + y, 0)
def src_pad_mask(x, device):
    src_mask =  (x != 0).type(torch.int32)#.view(BATCH_SIZE, 1, 1, -1).to(device)
    src_mask = src_mask.unsqueeze(1).unsqueeze(2)
    return src_mask.to(device)

def tgt_tril_mask(x, device):# mask out subsequent positions
    x_pad_mask = (x !=0).unsqueeze(1).unsqueeze(3).type(torch.int32).to(device)
    y = torch.tril(torch.ones([x.size(1), x.size(1)])).type(torch.int32).to(device)
    tgt_m = x_pad_mask & y
    return tgt_m.to(device)
# device = torch.device('cuda', 0) if torch.cuda.is_available() else "cpu"
device =  torch.device("mps") if torch.backends.mps.is_available() else "cpu"
print(device)
# torch.distributed.init_process_group(backend = 'nccl')
# local_rank = torch.distributed.get_rank()
# torch.cuda.set_device(local_rank)
# device = torch.device("cuda", local_rank)
def train():
    num_epochs = 700
    dataset, data_train = prepare_data(Train_file, device)
    # train_sampler = DistributedSampler(data_train, rank=local_rank, num_replicas=4, shuffle=False)
    # train_loader = DataLoader(data_train, sampler=train_sampler, batch_size=1)
    src_itow = [w for w, _ in sorted(src_vocab.items(), key = lambda x: x[1])]
    tgt_itow = [w for w, _ in sorted(tgt_vocab.items(), key = lambda x: x[1])]
    enc = encoder(len(src_vocab), device).to(device)
    dec = decoder(len(tgt_vocab), device).to(device)
    # enc = DDP(enc, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    # dec = DDP(dec, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    enc_optim = torch.optim.Adam(enc.parameters(), lr=0.000009)
    dec_optim = torch.optim.Adam(dec.parameters(), lr=0.000009)
    print("training model...")
    losses_1 = []
    losses_2 = []
    max_acc = 0
    maxAccEpochId = 0
    accuracies = []
    chunkend_losses = []
    for epoch in range(num_epochs):
        for index, (x, y) in enumerate(data_train):
            # x = x.squeeze(1)
            # y = y.squeeze(1)
            enc.train()
            dec.train()
            plot_data = []
            enc_optim.zero_grad()
            dec_optim.zero_grad()
            pad_mask = src_pad_mask(x, device)
            enc_out = enc(x, pad_mask)
            loss = 0
            tgt_indexes = [SOS_IDX]
            for t in range(y.size(1)):
                dec_in = torch.LongTensor(tgt_indexes).unsqueeze(0).to(device)
                dec_out = dec(enc_out, dec_in, pad_mask)
                loss += F.nll_loss(dec_out, y[:, t], size_average=False, ignore_index=PAD_IDX)
                form_id = dec_out.argmax().item()
                tgt_indexes.append(form_id)
           # dec_in = torch.LongTensor([SOS_IDX] * BATCH_SIZE).unsqueeze(1).to(device)
           # for t in range(y.size(1)):
           #     dec_out = dec(enc_out, dec_in, pad_mask)
           #     loss += F.nll_loss(dec_out, y[:, t], size_average=False, ignore_index=PAD_IDX)
           #     dec_in = torch.cat((dec_in, y[:, t].unsqueeze(1)), 1)  # teacher forcing

            # loss /= y.data.gt(0).sum().float() # divide by the number of unpadded tokens
            losses_1.append(loss.item())#cpu().detach().numpy())
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            enc_optim.step()
            dec_optim.step()
            # if local_rank == 0:
            if index % 50 == 0:
                plot_data.append(np.mean(losses_1[epoch * len(data_train) + index - 50:]))
                chunkend_losses.append(np.mean(plot_data))
                print("Epoch: {} Index: {} Loss: {}".format(epoch + 1, index, np.mean(losses_1)))
            # torch.distributed.barrier()
        # loss_sum /= len(data)
        dataset_test, data_test = prepare_data(Test_file, device)
        #test_sampler = DistributedSampler(data_test, rank=local_rank, num_replicas=3, shuffle=False)
        #test_loader = DataLoader(data_test, sampler=test_sampler, batch_size=1)
        print("Predicting..")
        correct = 0
        with torch.no_grad():
            for index, (x, y) in enumerate(data_test):
                #x = x.squeeze(1)
                #y = y.squeeze(1)
                enc.eval()
                dec.eval()
                pred = []
                src_m = src_pad_mask(x, device)
                enc_out = enc(x, src_m)
                tgt_indexes = [SOS_IDX]
                # dec_in = LongTensor([SOS_IDX] * BATCH_SIZE).unsqueeze(1)
                maximum = 100
                loss_2 = 0
                for i in range(y.size(1)):
                    dec_in = torch.LongTensor(tgt_indexes).unsqueeze(0).to(device)
                    dec_out = dec(enc_out, dec_in, src_m)
                    loss_2 += F.nll_loss(dec_out, y[:, t], size_average=False, ignore_index=PAD_IDX)
                    form_id = dec_out.argmax().item()
                    #print(dec_in)
                    tgt_indexes.append(form_id)
                    if form_id == EOS_IDX:
                        break
                    pred.append(form_id)
                losses_2.append(loss.item())#.cpu().detach().numpy())
                predicted = [tgt_itow[i] for i in pred if i != PAD_IDX]
                #print(predicted)
                #print(dataset_test[index][1])
                if len(dataset_test[index][1]) == len(predicted):
                    same = True
                    for g, p in zip(dataset_test[index][1], predicted):
                        if g != p:
                            same = False
                    if same:
                        correct += 1
                if index != 0:
                    if index % 50 == 0:
                        print("Epoch: {} Index: {} Val_Loss: {}".format(epoch + 1, index, np.mean(losses_2)))
        accuracy = 100*(correct/len(data_test))
        accuracies.append(accuracy)
        if accuracy > max_acc:
            max_acc = accuracy
            maxAccEpochId = epoch
        print("Accuracy: {} Max Accuracy {}".format(accuracy, max_acc))
    if not os.path.exists('out/Greedy_training1'):
        os.makedirs('out/Greedy_training1')


    file_name = "{}/{}".format('out/Greedy_training1', "accuracies")
    extra = "Maximum Accuracy {0:.2f} at epoch {1}".format(np.max(accuracies), maxAccEpochId)
    showPlot(accuracies, file_name, extra)
    file_name = "{}/{}".format('out/Greedy_training1', "all_losses")
    extra = "Mean Loss {0:.2f}".format(np.mean(chunkend_losses))
    showPlot(chunkend_losses, file_name, extra)

if __name__ == "__main__":
    train()
