"""Evaluate a language model on perturbed input in the restricted context setting."""

import argparse
import time, os
import math
import numpy as np
import csv
import gc
import _pickle as pickle

import torch
import torch.nn as nn
from torch.autograd import Variable

import data
import model

from utils_context import batchify_context, get_context_batch, get_vocab_all_pos
# this is not a good practice, but making an exception in this case
from perturbations import *

parser = argparse.ArgumentParser(description='Test language model on perturbed inputs')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
# Consistent with Language Modeling code from Merity et al.
parser.add_argument('--cuda', action='store_false',
                    help='Using this flag turns off CUDA, default value set to True')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--logdir', type=str, default='./',
                    help='location to write per token log loss')
parser.add_argument('--use_test', action='store_true', default=False,
                    help='Run on test set')
parser.add_argument('--batch_size', type=int, default=50, metavar='N',
                    help='batch size')
parser.add_argument('--seq_len', type=int, default=300,
                    help='sequence length')
parser.add_argument('--max_seq_len', type=int, default=1000,
                    help='max context size')
parser.add_argument('--freq', type=int, default=108,
                    help='Frequent words cut-off, all words with corpus count > 800')
parser.add_argument('--rare', type=int, default=1986,
                    help='Rare words cut-off, all words with corpus count < 50')
parser.add_argument('--func', type=str,
                    help='random_drop_many, drop_pos, keep_pos, shuffle, shuffle_within_spans, reverse, reverse_within_spans, replace_target, replace_target_with_nearby_token, drop_target, replace_with_rand_seq')
parser.add_argument('--span', type=int, default=20,
                    help='For shuffle and reverse within spans')
parser.add_argument('--drop_or_replace_target_window', type=int, default=300,
                    help='window for drop or replace target experiments')
parser.add_argument('--n', type=float,
                    help='Fraction of tokens to drop, between 0 and 1')
# Specify a list
parser.add_argument('--pos', action='append', default=None,
                    help='Pos tags to drop. Sample usage is --pos NN --pos VB --pos JJ')
parser.add_argument('--use_range', action='append', default=None,
                    help='Use these values for the boundary loop, but first convert to ints. Sample usage is --use_range 5 --use_range 20 --use_range 100')
args = parser.parse_args()
print(args)

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print('WARNING: You have a CUDA device, so you should probably run without --cuda')
    else:
        torch.cuda.manual_seed(args.seed)

print('Load model from %s' % args.checkpoint)
start = time.time()
model = torch.load(args.checkpoint)
print('[%.1f s]' % (time.time() - start))

if args.cuda:
    model.cuda()
else:
    model.cpu()

corpus = data.Corpus(args.data)
print('Built corpus')
pos_datafile = os.path.dirname(args.data if args.data.endswith('/') else args.data+'/')+'_pos/'
print(pos_datafile)
pos_corpus = data.Corpus(pos_datafile)
print('Built pos corpus')

if args.use_test:
    print('On test set!!!')
    data_ = batchify_context(corpus.test, args)
    pos_data = batchify_context(pos_corpus.test, args)
else:
    print('On validation set!!!')
    data_ = batchify_context(corpus.valid, args)
    pos_data = batchify_context(pos_corpus.valid, args)

print('Made batches')

#criterion = nn.CrossEntropyLoss(size_average=False)

def evaluate(data_source, pdata_source, boundary, func, args, pos2vocab):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_loss = 0
    n = 0
    total_len = 0
    save_all_losses = []

    ntokens = len(corpus.dictionary)
    pos_dict = pos_corpus.dictionary

    nbatches = (data_source.size(0) - args.seq_len) // args.batch_size
    examples_to_ignore = args.max_seq_len - args.seq_len
    print('Number of examples to ignore: %d' % examples_to_ignore)
    batches_to_ignore = examples_to_ignore // args.batch_size

    print('Batches: %d' % (nbatches-batches_to_ignore))

    for i in range(batches_to_ignore, nbatches):
        hidden = model.init_hidden(args.batch_size)

        data, targets = get_context_batch(data_source, i, args)
        if n == 0:
            print('First word: %s' % (corpus.dictionary.idx2word[data.data[-1,0]]))
        pdata, ptargets = get_context_batch(pdata_source, i, args)
        data = perturb_data(data.transpose(1,0), pdata.transpose(1,0), boundary, func, args, pos_dict, targets[-1].data, ptargets[-1].data, pos2vocab)

        output, _ = model(data, hidden)
        total_len += data.data.shape[0]*data.data.shape[1]
        output = nn.functional.log_softmax(output.permute(2,1,0)).permute(2,1,0)

        output_ = output[-1].data
        targets_ = targets[-1].data
        targets_ = targets_.unsqueeze(dim=1)

        CELoss = torch.gather(output_, dim=1, index=targets_).squeeze()
        CELoss = -1*CELoss
        save_all_losses += CELoss.tolist()
        loss = torch.sum(CELoss)

        total_loss += loss
        n += targets_.shape[0]

        if (n % 20000) == 0:
            print('Processed %d examples' % n)

        del targets, hidden

    print('Last word: %s' % (corpus.dictionary.idx2word[data.data[-1, -1]]))
    print('total:', n)
    print('=' * 89)
    print('Average Sequence Length post changes  %.4f' % (total_len / n))
    print('=' * 89)
    return total_loss / n, save_all_losses


def perturb_data(data, pdata, boundary, func, args, pos_dict, targets, ptargets, pos2vocab):
    if 'pos' in args.func:
        return func(data, pdata, boundary, args, pos_dict)
    elif args.func in ['replace_target', 'drop_target', 'replace_target_with_nearby_token']:
        return func(data, ptargets, args, targets, pos2vocab, corpus, pos_dict)
    elif args.func == 'replace_with_rand_seq':
        return func(data, boundary, args, corpus)
    return func(data, boundary, args)


# The length of each sequence is altered differently, so examples cannot be batched.
if args.func in ['drop_pos', 'keep_pos','drop_target']:
    assert(args.batch_size==1)

# Running with different boundary values to extract a trend.
if not args.use_range:
    looprange = [1, 5, 10, 15, 20, 30, 50, 100, 200]
else:
    looprange = [int(r) for r in args.use_range]

# For logging per token scores
prefix = args.func+'.per_tok_scores.'

# pos2vocab is a map containing all words with the given POS tag in a list sorted by frequency
if args.func in ['replace_target', 'replace_target_with_nearby_token', 'drop_target']:
    pos2vocab = get_vocab_all_pos(os.path.join(pos_datafile, 'test.txt' if args.use_test else 'valid.txt'),
            corpus.dictionary)
    looprange = [-1]
    prefix += str(args.drop_or_replace_target_window) + '.'
else:
    pos2vocab = None

print(looprange)

for boundary in looprange:
    if boundary == 0: # boundary 0 is not a logical choice here.
        continue
    log_msg, cfunc, res_label = pick_perturbation(args, boundary)
    print(log_msg)
    loss, all_losses = evaluate(data_, pos_data, boundary, cfunc, args, pos2vocab)
    res = [res_label, loss, math.exp(loss)]
    print(res)
    print('-' * 89)

    with open(os.path.join(args.logdir, prefix+str(boundary)), 'wb') as f:
        pickle.dump(res_label, f)
        pickle.dump(all_losses, f)
