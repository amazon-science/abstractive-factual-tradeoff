#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import click
import click_log
from logger import logger
# logger = logging.getLogger(__name__)
import json
from functools import partial
import tarfile
import tempfile
from load_datasets import load_cnn_dailymail, load_multi_news, \
    load_xsum, verify_checksums, merge_collect_files

# Show defaults in help messages
click.option = partial(click.option, show_default=True)


def unpack(tar_filename, output_dir):
    """Unpacks/loads the ConstraintsFact and ModelsFact datasets"""
    logger.info(f'Unpacking {tar_filename} to directory {output_dir}')
    os.makedirs(output_dir, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        with tarfile.open(tar_filename) as f:
            f.extractall(tmpdir)
        load_cnn_dailymail(f'{tmpdir}/cnn_dailymail',
                           f'{output_dir}/cnn_dailymail')
        load_xsum(f'{tmpdir}/xsum', f'{output_dir}/xsum')
        # Multi-News part only exists for the ConstraintsFact dataset,
        # but not for ModelsFact:
        is_constraints_fact_dataset = os.path.exists(f'{tmpdir}/multi_news_500')
        if is_constraints_fact_dataset:
            load_multi_news(f'{tmpdir}/multi_news_500',
                            f'{output_dir}/multi_news_500')
            load_multi_news(f'{tmpdir}/multi_news_800',
                            f'{output_dir}/multi_news_800')
        with open(f'{tmpdir}/checksums.json') as f:
            verify_checksums(output_dir, json.load(f).items())
            logger.info('Checksums verification: PASS')
        if is_constraints_fact_dataset:
            collect_files = [f'{output_dir}/cnn_dailymail/collect.json',
                             f'{output_dir}/multi_news_500/collect.json',
                             f'{output_dir}/multi_news_800/collect.json',
                             f'{output_dir}/xsum/collect.json']
        else:  # models_fact dataset:
            collect_files = [f'{output_dir}/cnn_dailymail/collect.json',
                             f'{output_dir}/xsum/collect.json']
        data = merge_collect_files(collect_files)
        for f in collect_files:
            os.remove(f)
        os.remove(f'{output_dir}/xsum/test.combined')
        with open(f'{output_dir}/data.jsonl', 'w') as f:
            for d in data:
                print(json.dumps(d), file=f)


@click.command()
@click_log.simple_verbosity_option(logger)
@click.argument('tar_filename', type=str)
def run(tar_filename):
    """This script loads the ConstraintsFact and ModelsFact datasets by
getting the document content (i.e., the source news articles) from the
respective sources (Huggingface cnn_dailymail, multi_news, xsum) and
inserting them into our dataset.

    """
    output_dir = tar_filename.replace('.tar.gz', '')
    unpack(tar_filename, output_dir)


if __name__ == '__main__':
    click_log.basic_config(logger)
    run()
