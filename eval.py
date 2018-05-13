"""Evaluate a language model."""

import argparse
import time, os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import _pickle as pickle

import data
import model

from utils import batchify, get_batch, repackage_hidden

parser = argparse.ArgumentParser(description='Evaluate Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--start_token', type=int, default=1000,
                    help='token where loss calculation ends')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--logfile', type=str, default='./',
                    help='location to write per token ppl')
parser.add_argument('--use_test', action='store_true', default=False,
                    help='Run on test set')
args = parser.parse_args()
print(args)

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

corpus = data.Corpus(args.data)
print('Built corpus')

seq_len = 100
eval_batch_size = 1
if args.use_test:
    print('On test set!!!')
    data_ = batchify(corpus.test, eval_batch_size, args)
else:
    print('On validation set!!!')
    data_ = batchify(corpus.valid, eval_batch_size, args)

print('Load model from %s' % args.checkpoint)
start = time.time()
model = torch.load(args.checkpoint)
print('[%.1f s]' % (time.time() - start))

if args.cuda:
    model.cuda()
else:
    model.cpu()

#criterion = nn.CrossEntropyLoss()

def evaluate(data_source, batch_size, seq_len):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_loss = 0
    tokens = 0
    n = 0
    save_all_losses = []

    ntokens = len(corpus.dictionary)

    hidden = model.init_hidden(batch_size)

    for i in range(0, data_source.size(0) - 1, seq_len):
        tokens += seq_len
        data, targets = get_batch(data_source, i, args, evaluation=True, seq_len=seq_len)
        output, hidden = model(data, hidden)
        output = nn.functional.log_softmax(output.permute(2,1,0)).permute(2,1,0)
        targets = targets.view(data.data.shape[0], batch_size, -1)
        CELoss = torch.gather(output.data, dim=2, index=targets.data).squeeze()
        CELoss = -1*CELoss
        if tokens < args.start_token: continue # We are not ready to accumulate error yet
        elif tokens >= args.start_token and tokens-seq_len < args.start_token:
            data.data = data.data[-(tokens-args.start_token+1):]
            CELoss = CELoss[-(tokens-args.start_token+1):]
            print('First word: %s' % (corpus.dictionary.idx2word[data.data[-(tokens-args.start_token+1),0]]))
        total_loss += torch.sum(CELoss)
        n += data.size(0)
        save_all_losses += CELoss.tolist()
        hidden = repackage_hidden(hidden)
    print('total: %d' % n)
    print('Last word: %s' % (corpus.dictionary.idx2word[data.data[-1,0]]))
    return total_loss / float(n), save_all_losses

loss, all_losses = evaluate(data_, eval_batch_size, seq_len)

print('=' * 89)
print('| loss {:5.5f} | ppl {:8.5f}'.format(
    loss, math.exp(loss)))
print('=' * 89)

with open(args.logfile, 'wb') as f:
    pickle.dump(seq_len, f)
    pickle.dump(all_losses, f)
