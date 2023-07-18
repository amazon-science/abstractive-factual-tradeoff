#! /usr/bin/env python

"""
This script produces summary-level factCC prediction scores by aggregating sentence-level scores
:param prediction_score: a file containing sentence-level factCC prediction scores
:param input: a jsonl file containing a list of summary sentences and their most relevant source article sentences,
produced by create_sentence_based_input.py
:return output: a file contianing summary-level factCC prediction scores
"""

import argparse
from collections import defaultdict
import json
import os
import sys

import numpy as np


def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--prediction_score', help="prediction scores of summary sentences by factcc", type=str)
    parser.add_argument('--input', help="input jsonl file containing summary id", type=str)
    parser.add_argument('--output', help="output directory storing sentence-level scores (min, max, and mean)")

    args = parser.parse_args(arguments)

    scores_sentence_level = np.loadtxt(args.prediction_score)
    scores_summary_level = defaultdict(list)
    with open(args.input, 'r') as input_file:
        for i, line in enumerate(input_file):
            # doc_id_pattern = 'doc.{summary_id}.{sentence_id}'
            summary_id = json.loads(line)['id'].split('.')[1]
            scores_summary_level[summary_id].append(scores_sentence_level[i])

    aggregated_summary_scores = []
    for _, scores in scores_summary_level.items():
        aggregated_summary_scores.append((np.min(scores), np.max(scores), np.mean(scores)))

    os.makedirs(args.output, exist_ok=True)

    with open(args.output + '/sentence_level_min.csv', 'w') as output_min_file:
        with open(args.output + '/sentence_level_max.csv', 'w') as output_max_file:
            with open(args.output + '/sentence_level_mean.csv', 'w') as output_mean_file:
                for min_score, max_score, mean_score in aggregated_summary_scores:
                    output_min_file.write(f'{min_score}\n')
                    output_max_file.write(f'{max_score}\n')
                    output_mean_file.write(f'{mean_score}\n')


if __name__ == '__main__':
    main(sys.argv[1:])
