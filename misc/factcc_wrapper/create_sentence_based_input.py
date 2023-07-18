#! /usr/bin/env python

"""
This script retrieves the most topk relevant sentences from an article given each summary sentence
using a sentence transformer model
:param input: a jsonl file containing a list of generated abstractive summary and its source articles
:param topk: number of sentences retrieved from each article. Default value is set to 10.
:return output: a jsonl file containing a list of summary sentences and their most relevant source article sentences
"""

import argparse
from functools import partial
import gc
import logging
from math import ceil
import torch.multiprocessing as mp
import json
from joblib import Parallel, delayed
import sys
from time import time
from tqdm import tqdm

import torch
import spacy
from spacy.util import minibatch

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SentenceSimilarityModel:
    def __init__(self, model=None, device="cpu"):
        self.model = SentenceTransformer(model, device=device) if model else \
            SentenceTransformer('distilbert-base-nli-stsb-mean-tokens', device=device)

    def embed(self, sentences):
        return self.model.encode(sentences)

    def similarity_matrix(self, sents1, sents2):
        '''
        Computes pairwise sentence similarities between the sentences in sents1 and the sentences in sents2
        '''
        logging.info('Computing {}x{} similarities'.format(len(sents1), len(sents2)))
        return cosine_similarity(self.embed(sents1), self.embed(sents2))


def retrieve_sents(sents_a, sents_tokens, sim, topk, max_tokens=500):
    '''
    Returns the top-k sentences of the articles
    '''
    topk = min(len(sents_a), topk)
    sim = torch.from_numpy(sim)
    topk_obj = torch.topk(sim, topk, sorted=False)
    indices = topk_obj.indices.numpy()
    values = topk_obj.values.numpy()

    result = []
    cur_tokens = 0
    total = 0
    for sent_token in sents_tokens:
        total += len(sent_token)

    for idx, val in sorted(zip(indices, values), key=lambda x: x[0]):
        if cur_tokens + len(sents_tokens[idx]) <= max_tokens:
            result.append(sents_a[idx])
            cur_tokens += len(sents_tokens[idx])
        else:
            result.append(' '.join(sents_tokens[idx][:max_tokens-cur_tokens]))
            break
    return result


def create_input(data, keys, topk, max_tokens, num_threads, batch_size):
    fields = keys.split(',')

    articles = []
    summaries = []
    for d in data:
        article, summary = d[fields[0]], d[fields[1]]
        articles.append(article)
        summaries.append(summary)

    # sentence segmentation and tokenization using Spacy
    start_time = time()
    nlp = spacy.load("en_core_web_lg", disable=["ner", 'tagger', 'parser', 'lemmatizer'])
    nlp.add_pipe('sentencizer')

    all_article_sents, all_article_tokens = process_text(nlp, articles, num_threads=num_threads, batch_size=batch_size)
    all_summary_sents, all_summary_tokens = process_text(nlp, summaries, num_threads=num_threads, batch_size=batch_size)
    end_time = time()
    delta = end_time - start_time
    logging.info(f'Took {delta:.3f} seconds for sentence segmentation and tokenization using spacy')

    # retrieval of article sentence using sentence transformer
    start_time = time()
    results = process_similarity(all_article_sents, all_article_tokens,
                                 all_summary_sents, all_summary_tokens,
                                 topk, max_tokens, num_threads)
    end_time = time()
    delta = end_time - start_time
    logging.info(f'Took {delta:.3f} seconds for sentence retrieval using sentence transformer')
    return results


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def process_similarity_batch(q, batch):
    '''
    :param batch_id:
    :param batch: batch_id, (doc_id, article_sents, summary_sents, article_tokens, summary_tokens, topk, max_tokens
    :return: json object storing id, claim, text, and label)
    '''
    results = []
    doc_id_pattern = 'doc.{}.{}'
    for doc_id, article_sents, summary_sents, article_tokens, summary_tokens, topk, max_tokens, model in batch:
        sim = model.similarity_matrix(summary_sents, article_sents)
        # Retrieve topk sentences for each summary sentence
        for sent_id, summary_sentence_tokens in enumerate(summary_tokens):
            retrieved_article_sents = \
                retrieve_sents(article_sents, article_tokens, sim[sent_id], topk,
                               max_tokens - len(summary_sentence_tokens))

            results.append({'id': doc_id_pattern.format(doc_id, sent_id),
                            'claim': summary_sents[sent_id],
                            'text': ' '.join(retrieved_article_sents),
                            'label': 'CORRECT'  # a dummy label
                            })
    q.put(results)


def process_similarity(all_article_sents, all_article_tokens, all_summary_sents, all_summary_tokens,
                       topk, max_tokens, num_threads):
    n_jobs = torch.cuda.device_count() or num_threads
    batch_size = ceil(len(all_article_sents) / n_jobs)

    models = [SentenceSimilarityModel(device=i) for i in range(n_jobs)]

    cur_count = 0
    cur_model_idx = 0
    arg_list = []
    for i, (article_sents, summary_sents, article_tokens, summary_tokens) in \
            enumerate(zip(all_article_sents, all_summary_sents, all_article_tokens, all_summary_tokens)):
        if cur_count == batch_size:
            cur_model_idx += 1
            cur_count = 0
        arg_list.append((i, article_sents, summary_sents, article_tokens, summary_tokens,
                         topk, max_tokens, models[cur_model_idx]))
        cur_count += 1

    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    p = ctx.Process(target=process_similarity_batch, args=(q, arg_list))
    p.start()
    results = q.get()
    p.join()
    return results


def process_nlp_batch(nlp, batch_id, batch):
    '''
    :param nlp: spacy model
    :param batch_id: batch_id
    :param batch: doc_id, text_list
    :return:
    '''
    texts = [text_list for doc_id, text_list in batch]
    return [process_doc(doc) for doc in nlp.pipe(texts)]


def process_doc(doc):
    sentences = [s.text for s in doc.sents]
    tokens = [[t.text for t in s] for s in doc.sents]
    return sentences, tokens


def process_text(nlp, text_list, num_threads, batch_size):
    executor = Parallel(n_jobs=num_threads, backend="multiprocessing",
                        prefer="processes", verbose=10 * (logging.DEBUG == logging.INFO))

    arg_list = []
    for i, text in enumerate(text_list):
        arg_list.append((i, text))

    batches = minibatch(arg_list, batch_size)
    do = delayed(partial(process_nlp_batch, nlp))
    tasks = (do(i, batch) for i, batch in tqdm(enumerate(batches), total=ceil(len(arg_list)/batch_size)))

    all_sentences = []
    all_tokens = []
    for batch in executor(tasks):
        for sentences, tokens in batch:
            all_sentences.append(sentences)
            all_tokens.append(tokens)

    gc.collect()
    return all_sentences, all_tokens


def read_file(file_path):
    data = []
    with open(file_path, 'r') as input_file:
        for line in input_file:
            line = line.strip()
            data.append(json.loads(line))
    return data


def write_file(data, file_path):
    with open(file_path, 'w') as output_file:
        for d in data:
            output_file.write(json.dumps(d) + '\n')


def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', '--input', help="Input", type=str)
    parser.add_argument('-o', '--output', help="Output", type=str)
    parser.add_argument('-k', '--keys', help="JSON keys for article,summary strings in the input",
                        type=str, default='article,summary')
    parser.add_argument('--sep', help="Separator between articles in multi-doc",
                        type=str, default=' story_separator_special_tag ')
    parser.add_argument('--topk', default=10, type=int, help="number of retrieved article sentences"
                                                            "per summary sentence")
    parser.add_argument('--max_tokens', default=500, type=int, help="minimum number of tokens")
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=500)

    args = parser.parse_args(arguments)

    data = read_file(args.input)
    results = create_input(data, args.keys, args.topk, args.max_tokens, args.threads, args.batch_size)
    write_file(results, args.output)


if __name__ == '__main__':
    main(sys.argv[1:])
