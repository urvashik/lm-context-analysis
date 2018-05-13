import torch
from torch.autograd import Variable
import numpy as np

def batchify_context(data, args):
    """Truncate corpus so remaining data can be split into batches evenly."""
    nbatch = data.size(0) // args.batch_size
    data = data.narrow(0, 0, nbatch * args.batch_size)

    print('Number of tokens after processing: %d' % data.size(0))

    if args.cuda:
        data = data.cuda()
    return data

def get_context_batch(source, i, args):
    """
    For restricted context size, the hidden state is not copied across targets, where almost every token serves as a target. The amount of data used depends on the sequence length.

    Examples of (context, target) pairs for the corpus "The cat sat on the mat to play with yarn" and sequence length 5:
        ("The cat sat on the", "mat")
        ("cat sat on the mat", "to")
        ("sat on the mat to", "play")
        ...
    """

    data_ = []
    target_ = []
    for j in range(args.batch_size):
        start = i * args.batch_size + j
        end = start + args.seq_len
        data_ += [source[start:end]]
        target_ += [source[start+1:end+1]]

    # No training, so volatile always True
    data = Variable(torch.stack(data_), volatile=True)
    target = Variable(torch.stack(target_))

    # sequence length x batch size for consistency with Merity et al.
    # Since each example corresponds to 1 target, only the last row of the targets variable are relevant, but passing the whole tensor for complete info.
    return data.transpose(1,0), target.transpose(1,0)

def get_vocab_all_pos(pos_datafile, corpus_dict):
    """
    Generate a map.
    Keys = POS tag
    Values = a list of words with that POS tag, sorted by frequency
    """
    pos_ = {}
    with open(pos_datafile, 'r') as f:
        for line in f:
            line = line.strip().split(' ') + ['<eos>_<eos>'] if len(line.strip()) > 0 else ['<eos>_<eos>']
            for word_pair in line:
                w, p = word_pair.split('_')
                if p not in pos_:
                    pos_[p] = {}
                token_id = corpus_dict.word2idx[w]
                pos_[p][token_id] = corpus_dict.counter[token_id]

    for tag in pos_:
        # sort dictionary by rank and throw away the frequencies
        pos_[tag] = sorted(pos_[tag], key=pos_[tag].get)

    return pos_
