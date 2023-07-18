#!/usr/bin/env python

"""Calculates ROUGE scores using pyrouge.

Adapted from:
https://github.com/nlpyang/PreSumm/blob/master/src/cal_rouge.py

"""

import argparse
import os
import time
import logging
from multiprocessing import Pool, cpu_count
import shutil
import sys
import json
import pyrouge
from pyrouge.utils.log import get_global_console_logger
from nltk import sent_tokenize
import tempfile

get_global_console_logger(logging.WARNING)
logging.basicConfig(level=logging.WARNING)

def print_split_text(text, ostream):
    """Splits by sentence to enable sentence-level ROUGE-L. files2rouge
    does that as well, but it requires pre-tokenized input as it
    splits on " .", see
    https://github.com/pltrdy/files2rouge/blob/master/files2rouge/utils.py#L36. Here
    we split by sentence using NLTK.

    """
    for sent in sent_tokenize(text):
        print(sent, file=ostream)

def process(data):
    candidates, references, pool_id = data
    cnt = len(candidates)
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    # tmp_dir = "rouge-tmp-{}-{}-{}".format(os.getpid(), current_time, pool_id)
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.mkdir(tmp_dir + "/candidate")
        os.mkdir(tmp_dir + "/reference")
        for i in range(cnt):
            if len(references[i]) < 1:
                continue
            with open(tmp_dir + "/candidate/cand.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                print_split_text(candidates[i], f)
            with open(tmp_dir + "/reference/ref.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                print_split_text(references[i], f)
        r = pyrouge.Rouge155()
        r.model_dir = tmp_dir + "/reference/"
        r.system_dir = tmp_dir + "/candidate/"
        r.model_filename_pattern = 'ref.#ID#.txt'
        r.system_filename_pattern = r'cand.(\d+).txt'
        # Default: https://github.com/bheinzerling/pyrouge/blob/master/pyrouge/Rouge155.py
        rouge_results = r.convert_and_evaluate(rouge_args='-e {} -c 95 -2 -1 -U -r 1000 -n 4 -w 1.2 -a -m'.format(r.data_dir))
        results_dict = r.output_to_dict(rouge_results)
    return results_dict


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def test_rouge(cand, ref, num_processes):
    """Calculate ROUGE scores of sequences passed as an iterator
       e.g. a list of str, an open file, StringIO or sys.stdin
    """
    candidates = [line.strip() for line in cand]
    references = [line.strip() for line in ref]

    assert len(candidates) == len(references)
    candidates_chunks = list(chunks(candidates, int(len(candidates) / num_processes)))
    references_chunks = list(chunks(references, int(len(references) / num_processes)))
    n_pool = len(candidates_chunks)
    arg_lst = []
    for i in range(n_pool):
        arg_lst.append((candidates_chunks[i], references_chunks[i], i))
    pool = Pool(n_pool)
    results = pool.map(process, arg_lst)
    final_results = {}
    for i,r in enumerate(results):
        for k in r:
            if(k not in final_results):
                final_results[k] = r[k] * len(candidates_chunks[i])
            else:
                final_results[k] += r[k] * len(candidates_chunks[i])
    for k in final_results:
        final_results[k] = final_results[k] / len(candidates)

    return final_results

def rouge_results_to_str(results_dict):
    return "ROUGE-F(1/2/3/4/L/SU4): {:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.2f}\nROUGE-P(1/2/3/4/L/SU4): {:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.2f}\nROUGE-R(1/2/3/4/L/SU4): {:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.2f}\n".format(
        results_dict["rouge_1_f_score"] * 100,
        results_dict["rouge_2_f_score"] * 100,
        results_dict["rouge_3_f_score"] * 100,
        results_dict["rouge_4_f_score"] * 100,
        results_dict["rouge_l_f_score"] * 100,
        results_dict["rouge_su*_f_score"] * 100,

        results_dict["rouge_1_precision"] * 100,
        results_dict["rouge_2_precision"] * 100,
        results_dict["rouge_3_precision"] * 100,
        results_dict["rouge_4_precision"] * 100,
        results_dict["rouge_l_precision"] * 100,
        results_dict["rouge_su*_precision"] * 100,

        results_dict["rouge_1_recall"] * 100,
        results_dict["rouge_2_recall"] * 100,
        results_dict["rouge_3_recall"] * 100,
        results_dict["rouge_4_recall"] * 100,
        results_dict["rouge_l_recall"] * 100,
        results_dict["rouge_su*_recall"] * 100
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=str, default="candidate.txt",
                        help='candidate file')
    parser.add_argument('-r', type=str, default="reference.txt",
                        help='reference file')
    parser.add_argument('-p', type=int, default=cpu_count()-1,
                        help='number of processes')
    parser.add_argument('-j', '--json', help="JSON oputput",
                        action='store_true')
    args = parser.parse_args()
    if args.c.upper() == "STDIN":
        candidates = sys.stdin
    else:
        candidates = open(args.c, encoding="utf-8")
    references = open(args.r, encoding="utf-8")

    results_dict = test_rouge(candidates, references,args.p)
    if args.json:
        print(json.dumps(results_dict))
    else:
        print(rouge_results_to_str(results_dict))
