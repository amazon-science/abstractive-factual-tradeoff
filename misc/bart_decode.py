#!/usr/bin/env python

"""Decode BART model, see
https://github.com/pytorch/fairseq/blob/master/examples/bart/README.summarization.md

"""

import os
import sys
import argparse
import logging
import re
import torch
from tqdm import tqdm
from warnings import warn
from torch.multiprocessing import Pool, set_start_method
set_start_method('spawn', force=True)
from functools import partial
import more_itertools as mit
from fairseq.models.bart import BARTModel
import nvgpu

logging.basicConfig(level=logging.WARNING)

def remove_newlines(s, replace=' '):
    '''Correctly removes newlines from string, incl. unicode newlines.

    '''
    return replace.join(s.splitlines())


def load_model(task, model_path):
    logging.info(f"Loading model {model_path}")
    model_dirname, model_fname = os.path.split(model_path)
    bart = BARTModel.from_pretrained(
        model_dirname,
        checkpoint_file=model_fname,
        data_name_or_path=task
    )
    return bart


def get_gpu_mem_usage(gpu_id):
    return nvgpu.gpu_info()[gpu_id]['mem_used_percent']/100.0

def get_gpu_mem_total(gpu_id):
    return nvgpu.gpu_info()[gpu_id]['mem_total']

def get_gpu_mem_used(gpu_id):
    return nvgpu.gpu_info()[gpu_id]['mem_used']


def run(data):
    task, model, source, beam, lenpen, max_len_a, max_len_b, min_len, \
        no_repeat_ngram_size, \
        extractive_penalty_fct, \
        batch_size, dynamic_batch, gpu_id = data
    batches = list(mit.chunked(source, batch_size))
    result = []
            
    with torch.cuda.device(gpu_id):

        bart = load_model(task, model)
        bart.cuda()
        bart.eval()
        bart.half()
        mem0 = get_gpu_mem_usage(gpu_id)

        def bart_sample(batch):
            with torch.no_grad():
                return bart.sample(batch, beam=beam, lenpen=lenpen,
                                   min_len=min_len, max_len_a=max_len_a, max_len_b=max_len_b,
                                   no_repeat_ngram_size=no_repeat_ngram_size,
                                   extractive_penalty_fct=extractive_penalty_fct)
        
        # Heuristically determine best batch size
        if dynamic_batch:
            batch = list(batches[0])
            if len(batch):
                bart_sample(batch)
                mem1 = get_gpu_mem_usage(gpu_id)
                batch_size = int( (.9 - mem0) * batch_size / (mem1 - mem0) )

        peak_mem = 0
        for i, batch_gen in enumerate(tqdm(batches)):
            batch = list(batch_gen)
            if len(batch):
                hypotheses_batch = bart_sample(batch)
                peak_mem = max(peak_mem, get_gpu_mem_used(gpu_id))
                for hypothesis in hypotheses_batch:
                    result.append(hypothesis)
                        
    return result, gpu_id, batch_size, peak_mem


def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', help="Input", type=str)
    parser.add_argument('-o', '--output', help="Output", type=str, default="summaries.txt")
    parser.add_argument('-m', '--model', help="Model path", type=str, required=True)
    parser.add_argument('--task', help="Task name", type=str, required=True)
    parser.add_argument('--gpus', help="Number of GPUs", type=int, default=len(nvgpu.available_gpus()))
    parser.add_argument('--beam', help="beam size", type=int, default=4)
    parser.add_argument('--lenpen', help="length penalty", type=float, default=2.0)
    parser.add_argument('--max-len-a', help="Max length for output: ax+b", type=int, default=0)
    parser.add_argument('--max-len-b', help="Max length for output: ax_b", type=int, default=140)
    parser.add_argument('--min-len', help="Min length for output", type=int, default=55)
    parser.add_argument('--no-repeat-ngram-size', help="Ngram blocking for repeated output ngrams", type=int, default=3)
    parser.add_argument('--no-dynamic-batch', help="Do not adapt batch size dynamically", action='store_false', dest="dynamic_batch")
    parser.add_argument('--extractive-penalty', help="Function to assign extractive penalty per token", type=str)
    parser.add_argument('-b', '--batch-size', help="Batch size per GPU", type=int, default=32)
    parser.add_argument('-d', '--debug', help="Print debugging statements",
                        action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.WARNING)
    parser.add_argument('-v', '--verbose', help="Verbose output",
                        action="store_const", dest="loglevel", const=logging.INFO)

    args = parser.parse_args(arguments)
    logging.basicConfig(level=args.loglevel)

    total_mem = get_gpu_mem_total(0)/1024

    inp = open(args.input, 'r', encoding="utf-8")
    output = open(args.output, 'w', encoding="utf-8")

    input_text = [line for line in inp]
    pool = Pool(args.gpus)
    args_list = []

    for i, input_split in enumerate(mit.divide(args.gpus, input_text)):
        args_list.append((args.task, args.model, list(input_split),
                          args.beam, args.lenpen, args.max_len_a, args.max_len_b, args.min_len,
                          args.no_repeat_ngram_size,
                          args.extractive_penalty,
                          args.batch_size, args.dynamic_batch, i))
    results = pool.map(run, args_list)
    for r,gpu_id,bsz,mem in results:
        for line in r:
            line = remove_newlines(line, '\\n')
            print(line, file=output)

        logging.info(f'Peak memory on GPU {gpu_id} with batch size {bsz}: {mem/1024:.2f} / {total_mem:.2f} GB')


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
