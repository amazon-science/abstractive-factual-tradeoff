#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import click
import logging
import click_log
import json
import datasets
# datasets.logging.set_verbosity_error()
from functools import partial
import re
import hashlib
import glob
import subprocess
from logger import logger

# Show defaults in help messages
click.option = partial(click.option, show_default=True)


def split_sentences(text, starts):
    """Splits text into sentences according to the passed sentence start
indexes.

    """
    result = []
    s = starts + [None]
    for i in range(len(s) - 1):
        sent = text[s[i]:s[i+1]].rstrip()
        result.append(sent)
    return result


def write_source_target_files(testset, input_dir, output_dir,
                              write_combined=False):
    """For XSum and CNN/DM, we reorder the samples that come from
HuggingFace/Datasets because we have used a different order.

    """
    with open(f'{input_dir}/order.json') as f:
        reorder = json.load(f)
    source_file = open(f'{output_dir}/test.source', 'w')
    target_file = open(f'{output_dir}/test.target', 'w')
    if write_combined:
        combined_file = open(f'{output_dir}/test.combined', 'w')
    for idx in reorder:
        doc, summary = testset[idx]
        print(doc, file=source_file)
        print(summary, file=target_file)
        if write_combined:  # used for xsum
            print(summary + ' ' + doc, file=combined_file)


def truncate_docs(s, lengths):
    '''Truncate the documents in the cluster to the specified lengths.'''
    split_sep = ' ||||| '
    docs = s.split(split_sep)
    join_sep = ' story_separator_special_tag '
    return join_sep.join([' '.join(doc.split()[:len])
                          for doc, len in zip(docs, lengths)])\
                   .replace(split_sep.lstrip(), join_sep.lstrip())


def clean_doc(s):
    return s.replace('\n', '\\n')


def multi_news_write_source_target_files(input_dir, output_dir,
                                         name='multi_news', version='1.0.0'):
    """Writes Multi-News source+target files."""
    logger.info('=> Getting Multi-News input documents from HuggingFace datasets repository')
    testset = datasets.load_dataset(name, version, split='test')
    N = len(testset)
    with open(f'{input_dir}/lengths.json') as f:
        lengths = json.load(f)
    with open(f'{output_dir}/test.source', 'w') as src, \
         open(f'{output_dir}/test.target', 'w') as trg:
        for idx, d in enumerate(testset):
            doc = truncate_docs(clean_doc(d['document']), lengths[idx])
            print(doc, file=src)
            print(d['summary'], file=trg)
    logger.info('')


def remove_newlines(s, replace=' '):
    '''Correctly removes newlines from string, incl. unicode newlines.

    '''
    return replace.join(s.splitlines())


def multi_news_clean_sent(s):
    """Reconstruct the cleaned sentences as they were shown to the
    annotators.

    """
    operations = [
        (r'\\n', r' -- '),
        (r'--\s+--', r'--'),
        (r'--\s+--', r'--'),
        (r'^\s*--', r''),
    ]
    for pattern, repl in operations:
        s = re.sub(pattern, repl, s)
    return s


def multi_news_write_collect_file(sent_starts,
                                  input_dir, output_dir,
                                  sep=' story_separator_special_tag '):
    with open(f'{output_dir}/test.source') as f:
        testset = [s.strip() for s in f.readlines()]
    collect_file = open(f'{input_dir}/collect.json')
    outfile = open(f'{output_dir}/collect.json', 'w')
    for linenum, line in enumerate(collect_file):
        line = line.strip()
        m = re.search(r'__(\d+)_(\d+)__', line)
        doc_id = int(m.group(1))
        docs_text = testset[doc_id]
        docs_sent_starts = sent_starts[doc_id]

        # Insert document content
        def myreplace_doc(m):
            assert doc_id == int(m.group(1))
            return json.dumps(remove_newlines(docs_text))
        line = re.sub(r'"__(\d+)__"', myreplace_doc, line)

        articles = remove_newlines(multi_news_clean_sent(docs_text)).split(sep)
        docs_sents = [s for i, d in enumerate(articles)
                      for s in split_sentences(d, docs_sent_starts[i])]

        def myreplace_sent(m):
            assert doc_id == int(m.group(1))
            idx = int(m.group(2))
            return docs_sents[idx].replace('"', '\\"').replace('\\ ', ' ')\
                                                      .replace("\\'", "'")
        line = re.sub(r'__(\d+)_(\d+)__', myreplace_sent, line)
        print(line, file=outfile)


def write_collect_file(documents, documents_orig, sent_starts, input_dir, output_dir):
    collect_file = open(f'{input_dir}/collect.json')
    outfile = open(f'{output_dir}/collect.json', 'w')
    for linenum, line in enumerate(collect_file):
        line = line.strip()
        m = re.search(r'__(\d+)_(\d+)__', line)
        doc_id = int(m.group(1))

        # Insert document content
        doc_text = documents[doc_id]
        doc_orig_text = documents_orig[doc_id]

        def myreplace_doc(m):
            assert doc_id == int(m.group(1))
            return json.dumps(doc_text)
        def myreplace_doc_orig(m):
            assert doc_id == int(m.group(1))
            return json.dumps(doc_orig_text)
        line = re.sub(r'(?<="document_full":)\s*"__(\d+)__"', myreplace_doc, line)
        line = re.sub(r'(?<="document_original":)\s*"__(\d+)__"', myreplace_doc_orig, line)

        # Insert document sentences
        docs_sents = split_sentences(doc_text, sent_starts[doc_id])

        def myreplace_sent(m):
            assert doc_id == int(m.group(1))
            idx = int(m.group(2))
            s = docs_sents[idx].replace('"', '\\"')\
                               .replace('\\ ', ' ')\
                               .replace("\\'", "'")\
                               .replace('\\y', 'y')
            return s
        line = re.sub(r'__(\d+)_(\d+)__', myreplace_sent, line)

        print(line, file=outfile)


def merge_collect_files(collect_files):
    result = []
    cnt = 0
    for fname in collect_files:
        with open(fname) as f:
            for line in f:
                d = json.loads(line)
                d['id'] = cnt
                result.append(d)
                cnt += 1
    return result


def verify_checksums(dirname, fname_checksum_pairs):
    all_correct = True
    for fname, expected in fname_checksum_pairs:
        with open(f'{dirname}/{fname}', 'rb') as f:
            data = f.read()
            checksum = hashlib.md5(data).hexdigest()
            if checksum != expected:
                all_correct = False
                raise Exception(
                    f'Incorrect checksum: {checksum} vs expected {expected} \
                    for file {fname}')
    return all_correct


def load_multi_news(input_dir, output_dir,
                    name='multi_news', version='1.0.0', data_split='test'):
    """Loads the Multi-News part of the dataset"""
    os.makedirs(output_dir, exist_ok=True)
    multi_news_write_source_target_files(input_dir, output_dir)
    with open(f'{input_dir}/sent_starts.json') as f:
        sent_starts = json.load(f)
    multi_news_write_collect_file(sent_starts, input_dir, output_dir)


@click.command()
@click.argument('input_dir', type=str)
@click.option('--output', 'output_dir', type=str, required=True,
              help='Output directory')
@click_log.simple_verbosity_option(logger)
def multi_news(input_dir, output_dir):
    """Loads the Multi-News part of the dataset"""
    load_multi_news(input_dir, output_dir)


def load_xsum(input_dir, output_dir):
    """Loads the XSum part of the dataset"""
    os.makedirs(output_dir, exist_ok=True)
    logger.info('=> Getting XSum input documents from HuggingFace datasets repository')
    testset_ = datasets.load_dataset('xsum', '1.2.0', split='test')
    testset = [(remove_newlines(x['document']).strip(), x['summary'])
               for x in testset_]
    write_source_target_files(testset, input_dir, output_dir,
                              write_combined=True)
    with open(f'{output_dir}/test.combined') as f:
        documents = [s.strip() for s in f.readlines()]
    with open(f'{output_dir}/test.source') as f:
        documents_orig = [s.strip() for s in f.readlines()]
    with open(f'{input_dir}/sent_starts.json') as f:
        sent_starts = json.load(f)
    write_collect_file(documents, documents_orig, sent_starts, input_dir, output_dir)
    apply_patches(input_dir, output_dir)
    logger.info('')


@click.command()
@click.argument('input_dir', type=str)
@click.option('--output', 'output_dir', type=str, required=True,
              help='Output directory')
@click_log.simple_verbosity_option(logger)
def xsum(input_dir, output_dir):
    """Loads the XSum part of the dataset"""
    load_xsum(input_dir, output_dir)


def clean_cnndm(s):
    return re.sub(r'^([A-z][a-z]+,? ){0,4}\(CNN\)', '', s)


def run_cmd(cmd):
    logger.debug(f"Running command: {cmd}")
    process = subprocess.Popen(cmd.split(),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    stdout_str = stdout.decode()
    stderr_str = stderr.decode()
    if stdout_str:
        logger.debug("Command stdout output: %s", stdout.decode())
    if stderr_str:
        logger.debug("Command stderr output: %s", stderr.decode())
        

def apply_patches(input_dir, output_dir):
    patches = glob.glob(f'{input_dir}/*.patch')
    for p in patches:
        f = p.replace(input_dir, output_dir).replace('.patch', '')
        cmd = f'patch -p1 {f} -i {p}'
        # subprocess.run(cmd, shell=True)
        run_cmd(cmd)


def load_cnn_dailymail(input_dir, output_dir):
    """Loads the CNN/DailyMail part of the dataset"""
    os.makedirs(output_dir, exist_ok=True)
    logger.info('=> Getting CNN/DM input articles from HuggingFace datasets repository')
    testset_ = datasets.load_dataset('cnn_dailymail', '1.0.0', split='test')
    testset = [(clean_cnndm(remove_newlines(x['article']).strip()),
                remove_newlines(x['highlights']))
               for x in testset_]
    write_source_target_files(testset, input_dir, output_dir)
    with open(f'{output_dir}/test.source') as f:
        documents = [s.strip() for s in f.readlines()]
    with open(f'{input_dir}/sent_starts.json') as f:
        sent_starts = json.load(f)
    write_collect_file(documents, documents, sent_starts, input_dir, output_dir)
    apply_patches(input_dir, output_dir)
    logger.info('')


@click.command()
@click.argument('input_dir', type=str)
@click.option('--output', 'output_dir', type=str, required=True,
              help='Output directory')
@click_log.simple_verbosity_option(logger)
def cnn_dailymail(input_dir, output_dir):
    """Loads the CNN/DailyMail part of the dataset"""
    load_cnn_dailymail(input_dir, output_dir)


@click.group()
def main():
    pass


main.add_command(multi_news)
main.add_command(xsum)
main.add_command(cnn_dailymail)


if __name__ == '__main__':
    click_log.basic_config(logger)
    main()
