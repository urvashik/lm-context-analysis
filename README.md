# How Neural Language Models Use Context

This repository contains code for experiments described in the ACL 2018 publication: [Sharp Nearby, Fuzzy Far Away: How Neural Language Models Use Context](arxiv link here). It was developed using the [AWD LSTM Language Model](https://github.com/salesforce/awd-lstm-lm/tree/bf0742cab41d8bf4cd817acfe7e5e0cbff4131ba) code (the older release in PyTorch v0.2.0) provided by Merity et al. 

Running this code requires the user to first train a language model. Then, they can evaluate it with restricted context and perturbed inputs to better understand how it is modeling nearby vs. long-range context. For further details on our experiments, we refer the reader to the paper.

If you use this code or results from our paper, please cite:

```
@article{khandelwal18context,
  title={{Sharp Nearby, Fuzzy Far Away: How Neural Language Models Use Context}},
  author={Khandelwal, Urvashi and He, He and Qi, Peng and Jurafsky, Dan},
  journal={Association of Computational Linguistics (ACL)},
  year={2018}
}
```

## Software Requirements

This code requires Python 3 and PyTorch v0.2.0. 

Using Anaconda, you can set up a conda environment by running `conda create -n lm_context python=3.4` and activating it with `source activate lm_context`. Then you can install PyTorch v0.2 with `conda install pytorch=0.2.0 -c soumith`.

## Training the Language Model

We have provided all the scripts from Merity et al. that are necessary to train and finetune language models for the Penn Treebank (PTB) and Wikitext-2 (Wiki) datasets.

+ First run `getdata.sh` to obtain the data.
+ Then, train the model as follows:
  + For PTB, `python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epoch 500 --save PTB.pt`
  + And for Wiki, `python main.py --epochs 750 --data data/wikitext-2 --save WT2.pt --dropouth 0.2 --seed 1882`
+ Finetune the model:
  + For PTB, `python finetune.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epoch 500 --save PTB.pt`
  + And for Wiki, `python finetune.py --epochs 750 --data data/wikitext-2 --save WT2.pt --dropouth 0.2 --seed 1882`

For further details on the model training instructions, we refer the user to the [original code](https://github.com/salesforce/awd-lstm-lm/tree/bf0742cab41d8bf4cd817acfe7e5e0cbff4131ba).

### Using your own language model

The analysis code provided in this repository is model-agnostic. It is widely usable with a number of different model classes.

If you would like to use your own language model, the analysis code expects a `init_hidden` function and a callable that returns a tuple of `outputs`, i.e. the output distributions, and the next `hidden` state. The sample evaluation script for the language model `eval.py` also expects all of the functionality in `utils.py`.

## Evaluation Script

`eval.py` allows the user to evaluate the language model in the standard infinite-context setting, i.e. the model is provided all previous tokens when estimating the distribution of the next one.

For PTB: `python -u eval.py --data data/penn/ --seed 141 --start_token 0 --checkpoint PTB.pt --logfile ptb_eval.pkl`

For Wiki: `python -u eval.py --data data/wikitext-2/ --seed 141 --start_token 0 --checkpoint WT2.pt --logfile wiki_eval.pkl`

**Note:** the `start_token` flag controls how many tokens to exclude from the beginning of the corpus. This is useful when trying to build a fair comparison between the infinite-context model and a restricted context model. For instance, if the context size is restricted to 300 tokens, the first 299 tokens cannot be included in the loss computation and this must be reflected in the loss for the infinite context model as well.

## Restricted context

`context_size.py` allows the user to evaluate their language model with a fixed size finite context. The `seq_len` flag defines how many tokens to use for context.

For PTB:  
`mkdir ptb_context_exps`  
`python -u context_size.py --data data/penn/ --checkpoint PTB.pt --seed 141 --batch_size 50 --seq_len 100 --max_seq_len 1000 --logdir ptb_context_exps/`

For Wiki:  
`mkdir wiki_context_exps`  
`python -u context_size.py --data data/wikitext-2/ --checkpoint WT2.pt --seed 1882 --batch_size 50 --seq_len 100 --max_seq_len 1000 --logdir wiki_context_exps/`

**Note:** You may need to use a smaller batch size for a larger sequence length and that would change the number of examples processed. Carefully pick the maximum sequence length first to process the same number of examples for each experiment.

## Perturbed inputs

You can find a list of functions supported by running `python perturb.py -h` and checking the `--func` flag. Sample commands for the reverse function:

For PTB:  
`mkdir ptb_reverse`  
`python -u perturb.py --data data/penn/ --checkpoint PTB.pt --seed 141 --logdir wiki_reverse --batch_size 50 --seq_len 300 --max_seq_len 1000 --freq 108 --rare 1986 --func reverse`

For Wiki:  
`mkdir wiki_reverse`  
`python -u perturb.py --data data/wikitext-2/ --checkpoint WT2.pt --seed 1882 --logdir wiki_reverse --batch_size 50 --seq_len 300 --max_seq_len 1000 --freq 189 --rare 4282 --func reverse`

## Neural Cache

`pointer.py` runs the neural cache at evaluation. It contains added support for saving all the losses to a file.

For PTB: `python pointer.py --data data/penn --save PTB.pt --lambdasm 0.1 --theta 1.0 --window 500 --bptt 5000 --logfile ptb_cache.pkl`

For Wiki: `python pointer.py --save WT2.pt --lambdasm 0.1279 --theta 0.662 --window 3785 --bptt 2000 --data data/wikitext-2 --logfile wiki_cache.pkl`

## TODOs

+ Add visualization code samples
+ Upgrade to PyTorch v0.3 (to reflect upgrade in Merity et al. code)
