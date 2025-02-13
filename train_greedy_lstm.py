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
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter("./out")

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
Whole_file = './data/whole.txt'
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
NUM_ENC_LAYERS = 2
NUM_DEC_LAYERS = 2
NUM_HEADS = 8 # number of heads
DK = EMBED_SIZE // NUM_HEADS # dimension of key
DV = EMBED_SIZE // NUM_HEADS # dimension of value
DROPOUT = 0.4
src_vocab = {PAD: PAD_IDX, EOS: EOS_IDX, SOS: SOS_IDX}
tgt_vocab = {PAD: PAD_IDX, EOS: EOS_IDX, SOS: SOS_IDX}
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
        src_tokens = src.split()#tokenize(src, "word")
        tgt_tokens = tgt.split()#tokenize(tgt, "word")
        for word in src_tokens:
            if word not in src_vocab:
                src_vocab[word] = len(src_vocab)
        for word in tgt_tokens:
            if word not in tgt_vocab:
                tgt_vocab[word] = len(tgt_vocab)
    fo.close()
    return src_vocab, tgt_vocab
#src_vocab = read_vocab(q_vocab)
#tgt_vocab = read_vocab(f_vocab)
src_vocab, tgt_vocab = creat_vocab(Train_file)
src_vocab, tgt_vocab = creat_vocab(Test_file)
print(len(src_vocab))
print(len(tgt_vocab))
def showPlot(points, fig_name, extra_info):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
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
device = torch.device('cpu')
dataset, data_train = prepare_data(Train_file, device)
for x, y in data_train:
    print(y.shape)
def train():
    num_epochs = 500
    dataset, data_train = prepare_data(Train_file, device)
    # train_sampler = DistributedSampler(data_train, rank=local_rank, num_replicas=4, shuffle=False)
    # train_loader = DataLoader(data_train, sampler=train_sampler, batch_size=1)
    src_itow = [w for w, _ in sorted(src_vocab.items(), key = lambda x: x[1])]
    tgt_itow = [w for w, _ in sorted(tgt_vocab.items(), key = lambda x: x[1])]
    enc = encoder(len(src_vocab), device).to(device)
    dec = decoder(len(tgt_vocab), device).to(device)
    # enc = DDP(enc, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    # dec = DDP(dec, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    enc_optim = torch.optim.Adam(enc.parameters(), lr=0.000006)
    dec_optim = torch.optim.Adam(dec.parameters(), lr=0.000006)
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
            tgt_mask = src_pad_mask(y, device)
            info = dec(enc_out)
            tgt_indexes = [SOS_IDX]
            for t in range(y.size(1)):
                dec_in = torch.LongTensor(tgt_indexes).unsqueeze(0).to(device)
                tgt_mask = src_pad_mask(dec_in, device)
                dec_out = dec(enc_out, dec_in, pad_mask, tgt_mask)
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
            # writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + index)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            enc_optim.step()
            dec_optim.step()
            if local_rank == 0:
                if index % 50 == 0:
                    plot_data.append(np.mean(losses_1[epoch * len(data_train) + index - 50:]))
                    chunkend_losses.append(np.mean(plot_data))
                    print("Epoch: {} Index: {} Loss: {}".format(epoch + 1, index, np.mean(losses_1)))
            torch.cuda.empty_cache()
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
                    tgt_m = src_pad_mask(dec_in, device)
                    dec_out = dec(enc_out, dec_in, src_m, tgt_m)
                    loss_2 += F.nll_loss(dec_out, y[:, t], size_average=False, ignore_index=PAD_IDX)
                    form_id = dec_out.argmax().item()
                    #print(dec_in)
                    tgt_indexes.append(form_id)
                    if form_id == EOS_IDX:
                        break
                    pred.append(form_id)
                losses_2.append(loss.item())#.cpu().detach().numpy())
                writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + index)
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

# if __name__ == "__main__":
    # train()
