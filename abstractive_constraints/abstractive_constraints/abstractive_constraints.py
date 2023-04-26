#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Decoding constraints which can penalize actions such as copying
source tokens.

"""

import os
import sys
import argparse
import logging
import json
from collections import defaultdict
import math
import re


# logging.basicConfig(level=logging.DEBUG)

class LengthFunction:
    def create(name, params=None):
        """
        >>> f = LengthFunction.create('linear')
        >>> [f(3), f(4)]
        [3, 4]
        >>> f = LengthFunction.create('log_exp', (2, 5))
        >>> [f(0), f(4), f(5)]
        [0.0, 0.64, 1.0]
        >>> f = LengthFunction.create('log_exp', (3, 5))
        >>> [f(0), f(4), f(5), f(6)]
        [0.0, 0.512, 1.0, 1.728]
        >>> f = LengthFunction.create('log_exp2', (3, 5))
        >>> [f(4), f(5)]
        [0.354891356446692, 0.6931471805599453]
        >>> f = LengthFunction.create('maxlen', (2))
        >>> [f(2), f(3)]
        [0.0, inf]
        >>> f = LengthFunction.create('maxlen_noinf', (2))
        >>> [f(2), f(3)]
        [0.0, 1000.0]
        """
        if name == "linear":
            return lambda x: x
        if name == "log_exp":
            (k, c) = params
            return lambda x: x**k / c**k
        if name == "log_exp_maxlen":
            (k, c, m) = params
            return lambda x: x**k / c**k if x <= m else math.inf
        if name == "log_exp_maxlen_noinf":
            # Here we avoid inf because it causes problems in k-best decoding
            (k, c, m) = params
            return lambda x: x**k / c**k if x <= m else 1000.0
        if name == "neg_log_exp_maxlen":
            (k, c, m) = params
            return lambda x: -x**k / c**k if x <= m else math.inf
        if name == "neg_log_exp":
            (k, c) = params
            return lambda x: -x**k / c**k
        if name == "log_exp2":
            (k, c) = params
            # If this is slow we can precompute or use LRU
            return lambda x: math.log(2.0**(x**k / c**k))
        if name == "neg_log_exp2":
            (k, c) = params
            return lambda x: -math.log(2.0**(x**k / c**k))
        if name == "none":
            return lambda x: 0.0
        if name == "maxlen":
            the_max = params
            return lambda x: 0.0 if x <= the_max else math.inf
        if name == "maxlen_noinf":
            the_max = params
            return lambda x: 0.0 if x <= the_max else 1000.0
        raise Exception('Unknown length function: "{}"'.format(name))

    def parse_description(description):
        """
        >>> LengthFunction.parse_description('linear()')
        ('linear', [])
        >>> LengthFunction.parse_description('log_exp(2,1)')
        ('log_exp', [2.0, 1.0])
        >>> LengthFunction.parse_description('log_exp(2,-1)')
        ('log_exp', [2.0, -1.0])
        >>> LengthFunction.parse_description('log_exp(2.5, 1.1)')
        ('log_exp', [2.5, 1.1])
        >>> LengthFunction.parse_description('maxlen(2)')
        ('maxlen', [2.0])
        >>> LengthFunction.parse_description('none()')
        ('none', [])
        """
        m = re.match(r'(.+?)\((.*?)\)', description)
        name = m.group(1)
        params_str = m.group(2)
        params = [float(i) for i in params_str.split(',') if len(i)]
        return name, params

    def create_from_description(description):
        """
        >>> f = LengthFunction.create_from_description('log_exp2(3, 5)')
        >>> [f(4), f(5)]
        [0.354891356446692, 0.6931471805599453]
        """
        name, params = LengthFunction.parse_description(description)
        return LengthFunction.create(name, params)


class IncrementalTokenMatcher():
    """
    Stores tokens "a" and incrementally matches tokens "b" against
    them.
    """
    def __init__(self, tokens=None,
                 token_len_fct=lambda _: 1.0):
        """
        Args:
            token_len_fct: Returns the length for the token. The
                default is 1.0 for each token, but it can be replaced
                by frequency-based weights (e.g., "the" doesn't count
                as much).
        """
        super().__init__()
        if tokens is not None:
            self.tokens = tokens
        self.token_len_fct = token_len_fct
        self.__query = []

    @property
    def tokens(self):
        return self.__tokens

    @property
    def query(self):
        return self.__query

    @tokens.setter
    def tokens(self, tokens):
        self.__tokens = tokens
        logging.debug('Setting tokens to {}'.format(tokens))
        self.precompute_tok_idx()
        self.reset_matches()

    def reset_matches(self):
        self.matches = dict()

    def precompute_tok_idx(self):
        """
        Precomputes the positions (idx) of each token in the input
        tokens.
        """
        self.tok_idx = defaultdict(lambda: [])
        for i, tok in enumerate(self.tokens):
            self.tok_idx[tok].append(i)

    def compute_matches(self, token):
        """
        Given the current query, computes the matches of the token.
        """
        logging.debug('compute_matches({})'.format(token))
        matches = dict()
        for idx in self.tok_idx[token]:
            prev_matchlen = self.matches.get(idx - 1, 0)
            matches[idx] = prev_matchlen + self.token_len_fct(token)
        return matches

    def peek(self):
        """
        Computes the matches for all possible continuations of the
        current output sequence.
        """
        d = {t: self.compute_matches(t) for t in self.tok_idx.keys()}
        return {k: v for k, v in d.items() if len(v)}

    def replay_query(self, query):
        self.reset_matches()
        self.__query = []
        for token in query:
            self.increment_query(token)

    def increment_query(self, token):
        """
        Increments the current query by one token.
        """
        self.__query.append(token)
        self.matches = self.compute_matches(token)
        return self.matches


class ExtractiveLengthPenalty(IncrementalTokenMatcher):
    def __init__(self, input_tokens,
                 token_len_fct=lambda _: 1.0,
                 penalty_fct=lambda x: x):
        super().__init__(input_tokens, token_len_fct=token_len_fct)
        self.penalty_fct = penalty_fct

    def compute_matches(self, token):
        matches = dict()
        for idx in self.tok_idx[token]:
            prev_matchlen = sum(self.matches.get(idx - 1, (0,0)))
            matches[idx] = (prev_matchlen, self.token_len_fct(token))
        return matches

    def peek(self):
        return {k: {i: self.penalty_fct(sum(matchlen)) - self.penalty_fct(matchlen[0])
                    for i, matchlen in v.items()}
                for k,v in super().peek().items()}

    def increment_query(self, token):
        penalties = [self.penalty_fct(sum(matchlen)) - self.penalty_fct(matchlen[0])
                     for matchlen in super().increment_query(token).values()]
        max_penalty = max(penalties) if len(penalties) else 0.0
        return max_penalty
