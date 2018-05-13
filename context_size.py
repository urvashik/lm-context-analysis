"""Evaluate a language model with restricted context, as opposed to infinite context."""

import argparse
import time, os
import math
import numpy as np
import csv
import _pickle as pickle

import torch
import torch.nn as nn
from torch.autograd import Variable

import data
import model

from utils_context import batchify_context, get_context_batch

parser = argparse.ArgumentParser(description='Restrict context size provided to language model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
# Consistent with Language Modeling code from Merity et al.
parser.add_argument('--cuda', action='store_false',
                    help='Using this flag turns off CUDA, default value set to True')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--batch_size', type=int, default=50, metavar='N',
                    help='batch size')
parser.add_argument('--seq_len', type=int, default=50,
                    help='starting context size')
parser.add_argument('--max_seq_len', type=int, default=1000,
                    help='Maximum possible sequence length to make all experiments start at the same example i.e. skip the same number of tokens at the start')
parser.add_argument('--logdir', type=str, default='./',
                    help='location to write per token log loss')
parser.add_argument('--use_test', action='store_true', default=False,
                    help='Run on test set')
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

if args.use_test:
    print('On test set!!!')
    data_ = batchify_context(corpus.test, args)
else:
    print('On validation set!!!')
    data_ = batchify_context(corpus.valid, args)
print('Made batches')

print('Context Size: %d' % args.seq_len)

# Not using the in-built loss function in order to compute per token losses.
# This was not supported in PyTorch 0.2.0.
#criterion = nn.CrossEntropyLoss(size_average=False)`

def evaluate(data_source, args):
    """Compute the log perplexities for the corpus, data_source."""
    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_loss = 0
    n = 0
    save_all_losses = []

    ntokens = len(corpus.dictionary)

    # Number of batches excludes the last seq_len tokens as start of context, since the target index would lie out of bounds.
    # Skip examples at the beginning to ensure the first target is the same for each experiment.
    # All experiments then operate on the same examples.
    # Select the max_seq_len as the largest context size you'd like to test.
    nbatches = (data_source.size(0) - args.seq_len) // args.batch_size
    examples_to_ignore = args.max_seq_len - args.seq_len
    print('Number of tokens to ignore: %d' % examples_to_ignore)
    batches_to_ignore = examples_to_ignore // args.batch_size

    print('Batches: %d' % (nbatches-batches_to_ignore))

    for i in range(batches_to_ignore, nbatches):
        # when using your own LM, ensure the init hidden function exists!
        hidden = model.init_hidden(args.batch_size)
        data, targets = get_context_batch(data_source, i, args)
        if n == 0:
            print('First word: %s' % (corpus.dictionary.idx2word[data.data[-1, 0]]))
        output, _ = model(data, hidden)
        # log probabilities for each vocab word for each token in each example
        output = nn.functional.log_softmax(output.permute(2,1,0)).permute(2,1,0)

        output_ = output[-1].data
        targets_ = targets[-1].data
        targets_ = targets_.unsqueeze(dim=1)
        if len(output_.shape) < 3: output_.unsqueeze(dim=0)

        CELoss = torch.gather(output_, dim=1, index=targets_).squeeze()
        CELoss = -1*CELoss
        save_all_losses += CELoss.tolist()
        loss = torch.sum(CELoss)

        total_loss += loss
        n += targets_.shape[0]

        if (n % 20000) == 0: print('Processed %d examples' % n)

        del output, targets, hidden

    print('Last word: %s' % (corpus.dictionary.idx2word[data.data[-1, -1]]))
    print('Total examples processed:', n)
    return total_loss / n, save_all_losses

loss, all_losses = evaluate(data_, args)
res = [args.seq_len, loss, math.exp(loss)]
print(res)

with open(os.path.join(args.logdir, 'per_token_scores_'+str(args.seq_len)), 'wb') as f:
    pickle.dump(args.seq_len, f)
    pickle.dump(all_losses, f)
