import torch
Train_file = 'train.txt'
Test_file = 'test.txt'
Whole_file = 'whole.txt'
PAD = "<PAD>" # padding
EOS = "<EOS>" # end of sequence
SOS = "<SOS>" # start of sequence
PAD_IDX = 0
EOS_IDX = 1
SOS_IDX = 2
def create_vocab(filename):
    src_vocab = {PAD: PAD_IDX, EOS: EOS_IDX, SOS: SOS_IDX}
    tgt_vocab = {PAD: PAD_IDX, EOS: EOS_IDX, SOS: SOS_IDX}
    with open(filename) as target:
        for line in target:
            src, tgt = line.strip().split('\t')
            src_tokens = src.split()
            tgt_tokens = tgt.split()
            for words in src_tokens:
                if words not in src_vocab:
                    src_vocab[words] = len(src_vocab)
            for words in tgt_tokens:
                if words not in tgt_vocab:
                    tgt_vocab[words] = len(tgt_vocab)
        return src_vocab, tgt_vocab
src_vocab, tgt_vocab = create_vocab(Whole_file)
def prepare_data(file):
    data = []
    dataset = []
    src_batch = []
    tgt_batch = []
    src_batch_len = 0
    tgt_batch_len = 0
    fo = open(file, "r")
    for line in fo:
        line = line.strip()
        src, tgt = line.split("\t")
        dataset.append((src.split(), tgt.split()))
        src_token = src.split()
        tgt_token = tgt.split()
        src_seq = []
        tgt_seq = []
        for word in src_token:
            if word not in src_vocab:
                src_vocab[word] = len(src_vocab)
            src_seq.append(str(src_vocab[word]))
        for word in tgt_token:
            if word not in tgt_vocab:
                tgt_vocab[word] = len(tgt_vocab)
            tgt_seq.append(str(tgt_vocab[word]))
            src_ind = [int(i) for i in src_seq] + [EOS_IDX]
            tgt_ind = [int(i) for i in tgt_seq] + [EOS_IDX]
        if len(src_ind) > src_batch_len:
            src_batch_len = len(src_ind)
        if len(tgt_ind) > tgt_batch_len:
            tgt_batch_len = len(tgt_ind)
        src_batch.append(src_ind)
        tgt_batch.append(tgt_ind)
        if len(src_batch) == 1:
            for seq in src_batch:
                seq.extend([PAD_IDX] * (src_batch_len - len(seq)))
            for seq in tgt_batch:
                seq.extend([PAD_IDX] * (tgt_batch_len - len(seq)))
            data.append((torch.LongTensor(src_batch), torch.LongTensor(tgt_batch)))
            src_batch = []
            tgt_batch = []
            src_batch_len = 0
            tgt_batch_len = 0
    return dataset, data
dataset, data = prepare_data(Test_file)
form = ['(', 'lambda', '$0', 'e', '(', 'loc:t', 'c0', '$0', ')', ')']
correct = 0.0
if len(form) == len(dataset[0][1]):
    same = True
    for g, p in zip(form, dataset[0][1]):
        if g != p:
            same = False
    if same:
        correct += 1
print(correct)