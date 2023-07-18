#! /usr/bin/env python

"""
This script calls factCC package to predict the factuality for each pair of summary sentence and
its relevant article sentences
:param data_dir: a directory containing a file named data-dev.jsonl storing a list of summary sentences
and their most relevant source article sentences
:param checkpoint_dir: a directory storing factCC model checkpoint
:param output_dir: output directory
:param visible_gpus: gpu ids used for running factCC
:return output: a file containing a list of factcc prediction scores
"""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import time

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from modeling.utils import (compute_metrics, convert_examples_to_features, output_modes, processors)
from modeling.run import MODEL_CLASSES, set_seed, load_and_cache_examples, make_model_input

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tqdm import tqdm


logger = logging.getLogger(__name__)


def evaluate(args, model, tokenizer):
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        os.makedirs(eval_output_dir, exist_ok=True)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)

        preds = None
        out_label_ids = None

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = make_model_input(args, batch)
                outputs = model(**inputs)

                logits_ix = 1 if args.model_type == "bert" else 7
                logits = outputs[logits_ix]

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        probabilities = torch.nn.functional.softmax(torch.Tensor(preds), dim=1)
        # 0 for CORRECT, 1 for INCORRECT
        probabilities_df = pd.DataFrame(probabilities[:, 0].numpy())
        probabilities_df.to_csv(os.path.join(eval_output_dir, "prediction_score.csv"), header=None, index=None)

    return results


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name: e.g., bert")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--checkpoint_dir", default=None, type=str, required=True,
                        help="The directory storing model checkpoints.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions will be written.")

    # Optional parameters
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--visible_gpus', default='-1', type=str, help="gpu ids")

    # parameters required for calling factCC, using default values
    parser.add_argument("--local_rank", type=int, default=-1,
                         help="For distributed training: local_rank")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    args = parser.parse_args()

    # setup gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.n_gpu = torch.cuda.device_count()

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    # Set seed
    set_seed(args)

    # Prepare task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    args.output_mode = output_modes[args.task_name]

    # Load pretrained model and tokenizer
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)

    # Inference
    model = model_class.from_pretrained(args.checkpoint_dir)

    if torch.cuda.device_count() > 1:
        logging.info(f"Use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model.to(args.device)
    evaluate(args, model, tokenizer)


if __name__ == "__main__":
    main()
