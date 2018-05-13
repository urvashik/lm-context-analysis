"""
A variety of perturbation functions for language model inputs.
Perturbations are usually applied to a substring within the input.
`boundary` defines the number of tokens nearest to the target that remain unperturbed.
Hence, the substring is defined as the first (args.seq_len - boundary) tokens

For example if the args.seq_len = 10, boundary is 4 and the function is reverse, the following transformation takes place:
    the cat sat on the mat to play with yarn ==> mat the on sat cat the to play with yarn

Note: While the conversion to numpy and back is inefficient, the numpy flexibility, in many cases, lends itself to a more readable implementation.
The code stands to be improved by not only converting the perturbations to operate in pytorch, but also by upgrading to the newest release of pytorch.
"""
# TODO: Convert functions to operate on pytorch tensors.

import numpy as np
import torch
from torch.autograd import Variable

def random_drop_many(data, boundary, args):
    """
    Drop (100*args.n)% of tokens from each substring, randomly sampled.
    """
    examples = []
    num_drop = math.floor((args.seq_len-boundary) * args.n)
    for example in range(data.data.shape[0]):
        drop_idxs = np.random.choice(args.seq_len-boundary, size=num_drop, replace=False)
        new_ex = np.delete(data.data[example].cpu().numpy(), drop_idxs).tolist()
        examples.append(new_ex)

    examples = [torch.cuda.LongTensor((ex)) for ex in examples]
    data = Variable(torch.stack(examples)).cuda()
    return data.transpose(1,0)

def drop_pos(data, pdata, boundary, args, pos_dict):
    """
    Drop all tokens, from the substring, with a part-of-speech tag that belongs to the list defined by args.pos.
    """
    examples = []
    for example in range(data.data.shape[0]):
        pcurr = pdata.data[example][:-boundary].cpu().numpy()
        drop_idxs = np.array([idx for idx in range(pcurr.shape[0]) if pos_dict.idx2word[pcurr[idx]] in args.pos], dtype=np.int32)
        new_ex = np.delete(data.data[example].cpu().numpy(), drop_idxs).tolist()
        examples.append(new_ex)

    examples = [torch.cuda.LongTensor((ex)) for ex in examples]
    data = Variable(torch.stack(examples)).cuda()
    return data.transpose(1,0)

def keep_pos(data, pdata, boundary, args, pos_dict):
    """
    Drop all tokens, from the substring, with a part-of-speech tag that does NOT belong to the list defined by args.pos.
    """
    examples = []
    for example in range(data.data.shape[0]):
        pcurr = pdata.data[example][:-boundary].cpu().numpy()
        idxs = set(list(range(args.seq_len-boundary)))
        keep_idxs = set([idx for idx in range(pcurr.shape[0]) if pos_dict.idx2word[pcurr[idx]] in args.pos])
        drop_idxs = np.array(sorted(list(idxs - keep_idxs)), dtype=np.int32)
        new_ex = np.delete(data.data[example].cpu().numpy(), drop_idxs).tolist()
        examples.append(new_ex)

    examples = [torch.cuda.LongTensor((ex)) for ex in examples]
    data = Variable(torch.stack(examples)).cuda()
    return data.transpose(1,0)

def shuffle(data, boundary, args):
    """
    Shuffle all tokens in the substring.
    """
    examples = []
    for example in range(data.data.shape[0]):
        new_ex = data.data[example].cpu().numpy()
        np.random.shuffle(new_ex[:-boundary])
        new_ex = new_ex.tolist()
        examples.append(new_ex)

    examples = [torch.cuda.LongTensor((ex)) for ex in examples]
    data = Variable(torch.stack(examples)).cuda()
    return data.transpose(1,0)

def shuffle_within_spans(data, boundary, args):
    """
    Shuffle the last args.span tokens of the substring.
    Note here that the substring is always the first (args.seq_len - boundary) tokens.
    So the tokens shuffled are [args.seq_len - boundary - args.span, args.seq_len - boundary)
    """
    examples = []
    for example in range(data.data.shape[0]):
        new_ex = data.data[example].cpu().numpy()
        edge = -boundary-args.span
        np.random.shuffle(new_ex[edge:-boundary])
        new_ex = new_ex.tolist()
        examples.append(new_ex)

    examples = [torch.cuda.LongTensor(ex) for ex in examples]
    data = Variable(torch.stack(examples)).cuda()
    return data.transpose(1,0)

def reverse(data, boundary, args):
    """
    Reverse the substring.
    """
    examples = []
    for example in range(data.data.shape[0]):
        new_ex = data.data[example].cpu().numpy()
        new_ex[:-boundary] = new_ex[:-boundary][::-1]
        new_ex = new_ex.tolist()
        examples.append(new_ex)

    examples = [torch.cuda.LongTensor((ex)) for ex in examples]
    data = Variable(torch.stack(examples)).cuda()
    return data.transpose(1,0)

def reverse_within_spans(data, boundary, args):
    """
    Reverse the last args.span tokens of the substring.
    Note here that the substring is always the first (args.seq_len - boundary) tokens.
    So the tokens reversed are [args.seq_len - boundary - args.span, args.seq_len - boundary)
    """
    examples = []
    for example in range(data.data.shape[0]):
        new_ex = data.data[example].cpu().numpy()
        edge = -boundary-args.span
        new_ex[edge:-boundary] = new_ex[edge:-boundary][::-1]
        new_ex = new_ex.tolist()
        examples.append(new_ex)

    examples = [torch.cuda.LongTensor(ex) for ex in examples]
    data = Variable(torch.stack(examples)).cuda()
    return data.transpose(1,0)

def replace_target(data, ptargets, args, targets, pos2vocab, corpus, pos_dict):
    """
    Replace occurrences of the target word in the context, with the <unk> token.
    args.drop_or_replace_target_window defines the number of tokens nearest to the target, to be perturbed.
    For example, if args.drop_or_replace_target_window = 50, and args.seq_len = 300, only the last 50 tokens are considered for perturbation.
    This is to facilitate analysis for whether the model copies from nearby context vs. from long-range context.
    """
    examples = []
    for example in range(data.data.shape[0]):
        new_ex1 = data.data[example].tolist()
        tag = pos_dict.idx2word[ptargets[example]]
        token_idx_in_list = pos2vocab[tag].index(targets[example])
        # This step is taken to remain comparable with replace_target_with_nearby_token.
        # Words that are not replaced in the other function are also not replaced in this case.
        # Eg: <eos> in Wikitext-2 is not replaced in either function.
        if token_idx_in_list == 0 and len(pos2vocab[tag]) == 1:
            replace_idx = pos2vocab[tag][token_idx_in_list]
        else:
            replace_idx = corpus.dictionary.word2idx['<unk>']
        new_ex = [i if i != targets[example] or k < (args.seq_len - args.drop_or_replace_target_window) else replace_idx for k, i in enumerate(new_ex1)]
        examples.append(new_ex)

    examples = [torch.cuda.LongTensor(ex) for ex in examples]
    data = Variable(torch.stack(examples)).cuda()
    return data.transpose(1,0)

def replace_target_with_nearby_token(data, ptargets, args, targets, pos2vocab, corpus, pos_dict):
    """
    Replace occurrences of the target word in the context, with the a token that has the same POS tag and is as closest to the target in terms of frequency.
    args.drop_or_replace_target_window defines the number of tokens nearest to the target, to be perturbed.
    For example, if args.drop_or_replace_target_window = 50, and args.seq_len = 300, only the last 50 tokens are considered for perturbation.
    This is to facilitate analysis for whether the model copies from nearby context vs. from long-range context.
    """
    examples = []
    for example in range(data.data.shape[0]):
        new_ex = data.data[example].tolist()
        tag = pos_dict.idx2word[ptargets[example]]
        token_idx_in_list = pos2vocab[tag].index(targets[example])
        if token_idx_in_list == 0:
            if len(pos2vocab[tag]) == 1:
                replace_idx = pos2vocab[tag][token_idx_in_list]
            else:
                replace_idx = pos2vocab[tag][token_idx_in_list+1]
        elif token_idx_in_list == len(pos2vocab[tag])-1:
            replace_idx = pos2vocab[tag][token_idx_in_list-1]
        else:
            # Find the token with closest frequency
            leftdiff = abs(corpus.dictionary.counter[pos2vocab[tag][token_idx_in_list]] - corpus.dictionary.counter[pos2vocab[tag][token_idx_in_list-1]])
            rightdiff = abs(corpus.dictionary.counter[pos2vocab[tag][token_idx_in_list]] - corpus.dictionary.counter[pos2vocab[tag][token_idx_in_list+1]])
            if leftdiff <= rightdiff:
                replace_idx = pos2vocab[tag][token_idx_in_list-1]
            else:
                replace_idx = pos2vocab[tag][token_idx_in_list+1]
        new_ex = [i if i != targets[example] or k < (args.seq_len - args.drop_or_replace_target_window) else replace_idx for k, i in enumerate(new_ex)]
        examples.append(new_ex)

    examples = [torch.cuda.LongTensor(ex) for ex in examples]
    data = Variable(torch.stack(examples)).cuda()
    return data.transpose(1,0)

def drop_target(data, ptargets, args, targets, pos2vocab, corpus, pos_dict):
    """
    Drop occurrences of the target word in the context.
    args.drop_or_replace_target_window defines the number of tokens nearest to the target, to be perturbed.
    For example, if args.drop_or_replace_target_window = 50, and args.seq_len = 300, only the last 50 tokens are considered for perturbation.
    This is to facilitate analysis for whether the model copies from nearby context vs. from long-range context.
    """
    examples = []
    for example in range(data.data.shape[0]):
        new_ex = data.data[example].tolist()
        tag = pos_dict.idx2word[ptargets[example]]
        token_idx_in_list = pos2vocab[tag].index(targets[example])
        # This step is taken to remain comparable with replace_target_with_nearby_token.
        # Words that are not replaced in the other function are also not replaced in this case.
        # Eg: <eos> in Wikitext-2 is not replaced in either function.
        if token_idx_in_list == 0 and len(pos2vocab[tag]) == 1:
            drop_idxs = []
        else:
            drop_idxs = [k for k, i in enumerate(new_ex) if i == targets[example] and k >= (args.seq_len - args.drop_or_replace_target_window)]
        new_ex = np.delete(np.array(new_ex), drop_idxs).tolist()
        examples.append(new_ex)

    examples = [torch.cuda.LongTensor(ex) for ex in examples]
    data = Variable(torch.stack(examples)).cuda()
    return data.transpose(1,0)

def replace_with_rand_seq(data, boundary, args, corpus):
    """
    Replace the substring with sequences drawn from the training set.
    """
    examples = []
    for example in range(data.data.shape[0]):
        rep_idx = np.random.randint(corpus.train.shape[0]-args.seq_len)
        data.data[example][:-boundary] = corpus.train[rep_idx:rep_idx+args.seq_len-boundary]

    return data.transpose(1,0)

def pick_perturbation(args, boundary):
    if args.func == 'random_drop_many':
        log_msg = 'Randomly dropping %.3f fraction of all the tokens %d tokens away from target' % (args.n, boundary)
        cfunc = random_drop_many
        res_label = boundary

    elif args.func == 'drop_pos':
        log_msg = 'Drop all words of [ %s ] pos tag %d tokens away from the target' % (' '.join(args.pos), boundary)
        cfunc = drop_pos
        res_label = boundary

    elif args.func == 'keep_pos':
        log_msg = 'Drop all words NOT of [ %s ] pos tag %d tokens away from the target' % (' '.join(args.pos), boundary)
        cfunc = keep_pos
        res_label = boundary

    elif args.func == 'shuffle':
        log_msg = 'Shuffle the context %d tokens away from the target' % boundary
        cfunc = shuffle
        res_label = boundary

    elif args.func == 'shuffle_within_spans':
        log_msg = 'Shuffle the context %d to %d tokens away from the target' % (boundary+args.span, boundary)
        cfunc = shuffle_within_spans
        res_label = boundary

    elif args.func == 'reverse':
        log_msg = 'Reverse the context %d tokens away from the target' % boundary
        cfunc = reverse
        res_label = boundary

    elif args.func == 'reverse_within_spans':
        log_msg = 'Reverse the context %d to %d tokens away from the target' % (boundary+args.span, boundary)
        cfunc = reverse_within_spans
        res_label = boundary

    elif args.func == 'replace_target':
        log_msg = 'Replace every occurrence of the target token in its context with <unk>, within %d tokens' % args.drop_or_replace_target_window
        cfunc = replace_target
        res_label = args.seq_len

    elif args.func == 'replace_target_with_nearby_token':
        log_msg = 'Replace every occurrence of the target token in its context with a token of the same POS tag and most similar frequency within %d tokens' % args.drop_or_replace_target_window
        cfunc = replace_target_with_nearby_token
        res_label = args.seq_len

    elif args.func == 'drop_target':
        log_msg = 'Drop every occurrence of the target token in its context within %d tokens' % args.drop_or_replace_target_window
        cfunc = drop_target
        res_label = args.seq_len

    elif args.func == 'replace_with_rand_seq':
        log_msg = 'Replace context with random sequence from training data %d tokens away from the target' % boundary
        cfunc = replace_with_rand_seq
        res_label = boundary

    else:
        raise ValueError('Perturbation not supported!')

    return log_msg, cfunc, res_label
