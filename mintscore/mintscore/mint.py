#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Calculates the Mint scores.

"""

import os
import sys
import argparse
import logging
import json
import spacy
from joblib import Parallel, delayed
from functools import partial
from multiprocessing import cpu_count
from spacy.util import minibatch
from scipy.stats import hmean
from mintlcs import lcs
import numpy as np

def ngrams_all(tokens, max_n=4):
    '''Returns the sets of 1-grams, 2-grams, ..., n-grams as a list of lists.
    >>> ngrams = ngrams_all(['a', 'b', 'c', 'd', 'e'])
    >>> [sorted(list(x)) for x in ngrams]
    [[('a',), ('b',), ('c',), ('d',), ('e',)], [('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'e')], [('a', 'b', 'c'), ('b', 'c', 'd'), ('c', 'd', 'e')], [('a', 'b', 'c', 'd'), ('b', 'c', 'd', 'e')]]
    >>> ngrams = ngrams_all(['a', 'b', 'c'], max_n=5)
    >>> [sorted(list(x)) for x in ngrams]
    [[('a',), ('b',), ('c',)], [('a', 'b'), ('b', 'c')], [('a', 'b', 'c')], [], []]
    '''
    mymax = min(len(tokens), max_n)
    offsets = [tokens[i:] for i in range(mymax)]
    result = [zip(*offsets[:i]) for i in range(1, mymax + 1)] + [()] * (max_n - mymax)
    return result


def ngram_overlaps(x_tok, y_tok, smooth=True):
    '''
    >>> ngram_overlaps(['a'], ['a', 'a', 'a', 'b', 'b', 'c'])
    [(2.3333333333333335, 6), (0.7777777777777778, 5), (0.25925925925925924, 4), (0.08641975308641975, 3)]
    >>> ngram_overlaps(['a'], ['a', 'a', 'a', 'b', 'b', 'c'], smooth=False)
    [(3, 6), (0, 5), (0, 4), (0, 3)]
    >>> ngram_overlaps('The Supreme Court Thursday reserved its decision on a batch of pleas that have raised questions'.split(), 'The Supreme Court Thursday reserved its verdict on a batch of pleas which have raised questions'.split())
    [(13.333333333333334, 16), (10.777777777777779, 15), (7.9259259259259265, 14), (5.3086419753086425, 13)]
    >>> ngram_overlaps('The Supreme Court Thursday reserved its decision on a batch of pleas that have raised questions'.split(), 'The Supreme Court Thursday reserved its verdict on a batch of pleas which have raised questions'.split(), smooth=False)
    [(14, 16), (11, 15), (8, 14), (5, 13)]
    >>> ngram_overlaps('The Supreme Court Thursday reserved its decision on a batch of pleas that have raised questions'.split(), 'Supreme Court has reserved a verdict on the batch of the pleas which have raised some questions'.split(),)
    [(8.666666666666666, 17), (3.888888888888889, 16), (1.2962962962962963, 15), (0.43209876543209874, 14)]
    >>> ngram_overlaps('The Supreme Court Thursday reserved its decision on a batch of pleas that have raised questions'.split(), 'Supreme Court has reserved a verdict on the batch of the pleas which have raised some questions'.split(), smooth=False)
    [(11, 17), (3, 16), (0, 15), (0, 14)]
    '''
    nx = ngrams_all(x_tok, 5)
    ny = ngrams_all(y_tok, 5)
    def count_match(x, y):
        return sum(1 for ngram in y if ngram in x)
    match_counts = [0] # zero-gram
    for (x, y) in zip(nx, ny):
        match_counts.append(count_match(set(x), list(y)))

    # Smoothing, Chen & Cherry (2005), method 5
    if smooth:
        match_counts[0] = match_counts[1] + 1
        for i in range(1, 5):
            match_counts[i] = (match_counts[i - 1] + match_counts[i] + match_counts[i + 1]) / 3.0

    lengths = range(len(y_tok), len(y_tok) - 4, -1)
    return list(zip(match_counts[1:5], lengths))


def mint(x_tok, y_tok, smooth=True):
    '''
    >>> mint(['a'], ['a', 'a', 'a', 'b', 'b', 'c'])
    {'p1': (2.3333333333333335, 6, 0.3888888888888889), 'p2': (0.7777777777777778, 5, 0.15555555555555556), 'p3': (0.25925925925925924, 4, 0.06481481481481481), 'p4': (0.08641975308641975, 3, 0.028806584362139915), 'lcsr': (1, 6, 0.16666666666666666), 'mint': 0.9232456140350878}
    >>> mint(['a', 'b', 'c', 'd'], ['a', 'b', 'c', 'd'], smooth=False)
    {'p1': (4, 4, 1.0), 'p2': (3, 3, 1.0), 'p3': (2, 2, 1.0), 'p4': (1, 1, 1.0), 'lcsr': (4, 4, 1.0), 'mint': 0.0}
    >>> mint(['a', 'b', 'c', 'd', 'e'], ['a', 'b', 'c', 'd', 'e'], smooth=False)
    {'p1': (5, 5, 1.0), 'p2': (4, 4, 1.0), 'p3': (3, 3, 1.0), 'p4': (2, 2, 1.0), 'lcsr': (5, 5, 1.0), 'mint': 0.0}
    >>> mint(['x'], ['a', 'b', 'c', 'd'])
    {'p1': (0.3333333333333333, 4, 0.08333333333333333), 'p2': (0.1111111111111111, 3, 0.037037037037037035), 'p3': (0.037037037037037035, 2, 0.018518518518518517), 'p4': (0.012345679012345678, 1, 0.012345679012345678), 'lcsr': (0, 4, 0.0), 'mint': 1.0}
    >>> mint('The Supreme Court Thursday reserved its decision on a batch of pleas that have raised questions'.split(), 'The Supreme Court Thursday reserved its verdict on a batch of pleas which have raised questions'.split())
    {'p1': (13.333333333333334, 16, 0.8333333333333334), 'p2': (10.777777777777779, 15, 0.7185185185185186), 'p3': (7.9259259259259265, 14, 0.5661375661375662), 'p4': (5.3086419753086425, 13, 0.40835707502374174), 'lcsr': (14, 16, 0.875), 'mint': 0.3710535235740673}
    >>> mint('The Supreme Court Thursday reserved its decision on a batch of pleas that have raised questions'.split(), 'Supreme Court has reserved a verdict on the batch of the pleas which have raised some questions'.split())
    {'p1': (8.666666666666666, 17, 0.5098039215686274), 'p2': (3.888888888888889, 16, 0.24305555555555555), 'p3': (1.2962962962962963, 15, 0.08641975308641975), 'p4': (0.43209876543209874, 14, 0.030864197530864196), 'lcsr': (10, 17, 0.5882352941176471), 'mint': 0.9033765130600977}
    '''
    overlaps = ngram_overlaps(x_tok, y_tok, smooth)
    lcs_ = lcs(x_tok, y_tok)
    lcsr = lcs_ / len(y_tok)
    mint = 1.0 if lcsr == 0.0 else 1.0 - hmean([num / denom for num, denom in overlaps] + [lcsr])
    d = {'p' + str(i + 1): (num, denom, num / denom) for i, (num, denom) in enumerate(overlaps)}
    d.update({'lcsr': (lcs_, len(y_tok), lcsr), 'mint': mint})
    return d
    

def process_batch(nlp, batch_id, batch):
    src, tgt = zip(*batch)
    src_docs, tgt_docs = nlp.pipe(src), nlp.pipe(tgt)
    def tokenize(doc):
        return [tok.lower_ for tok in doc]
    return [mint(tokenize(s), tokenize(t)) for s,t in zip(src_docs, tgt_docs)]


def main(arguments):

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('target', help="Target strings", type=str, nargs='?',
                        default='-')
    parser.add_argument('-s', '--source', help="Source strings", type=str, default='-')
    parser.add_argument('-o', '--output', help="Output", type=str, default='-')
    parser.add_argument('-c', '--compact', help="Compact output (prints mean and median)",
                        action='store_true')
    parser.add_argument('-t', '--test', help="Execute doctests and exit",
                        action='store_true')
    parser.add_argument('-d', '--debug', help="Print debugging statements",
                        action="store_const", dest="loglevel",
                        const=logging.DEBUG, default=logging.WARNING)
    parser.add_argument('-v', '--verbose', help="Verbose output",
                        action="store_const", dest="loglevel",
                        const=logging.INFO)
    parser.add_argument('-b', '--batch-size', help="Batch size", type=int, default=500)
    parser.add_argument('-j', '--jobs', help="Number of jobs", type=int, default=cpu_count()-1 or 1)

    args = parser.parse_args(arguments)
    logging.basicConfig(level=args.loglevel)

    if args.test:
        import doctest
        doctest.testmod(verbose=(args.loglevel <= logging.INFO))
        sys.exit()

    tgt_stream = sys.stdin if args.target == '-' else \
        open(args.target, 'r', encoding="utf-8")
    src_stream = open(args.source, 'r', encoding="utf-8")
    ostream = sys.stdout if args.output == '-' else \
        open(args.output, 'w', encoding="utf-8")

    def yield_input(src, tgt):
        for t in tgt:
            s = src.readline()
            yield s.strip(), t.strip()

    try:
        nlp = spacy.load("en_core_web_sm",
                         disable=["ner", "tagger", "parser", "lemmatizer"])
    except:
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm",
                         disable=["ner", "tagger", "parser", "lemmatizer"])
        
    batches = minibatch(yield_input(src_stream, tgt_stream), size=args.batch_size)
    executor = Parallel(n_jobs=args.jobs, backend="multiprocessing", prefer="processes", 
                        verbose=10*(args.loglevel==logging.INFO))
    do = delayed(partial(process_batch, nlp))
    tasks = (do(i, batch) for i, batch in enumerate(batches))
    mint = []
    for batch in executor(tasks):
        for result in batch:
            if args.compact:
                mint.append(result['mint'])
            else:
                print(json.dumps(result), file=ostream)
    if args.compact:
        d = {'mean': np.mean(mint), 'median': np.median(mint), 'variance': np.var(mint),
             'min': np.min(mint), 'max': np.max(mint), 'len': len(mint)}
        print(json.dumps(d), file=ostream)


def run():
    sys.exit(main(sys.argv[1:]))


if __name__ == '__main__':
    run()
