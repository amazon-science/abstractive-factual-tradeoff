import unittest

from abstractive_constraints.abstractive_constraints \
    import IncrementalTokenMatcher, LengthFunction, ExtractiveLengthPenalty

def sorted_dict(d):
    return dict(sorted(d.items()))

class TestIncrementalTokenMatcher(unittest.TestCase):

    def run_test(self, matcher, test):
        for token, expected in test:
            if token is not None:
                x = matcher.increment_query(token)
                assert(x == last_result[token]), 'Increment token: {}, result: {}, expected: {}'.format(token, sorted_dict(x), sorted_dict(last_result[token]))
            result = matcher.peek()
            assert(result == expected),'Peek result after token {}: {}, but expected {}'.format(token, sorted_dict(result), sorted_dict(expected))
            last_result = result

    def test1(self):
        matcher = IncrementalTokenMatcher([0, 2, 1, 2, 0, 1, 2])
        test = [
            # Read example: Initially, peek shows that token 1 would match a indices 2 and 5 with match len 1, etc.
            (None, {0: {0: 1.0, 4: 1.0}, 1: {2: 1.0, 5: 1.0}, 2: {1: 1.0, 3: 1.0, 6: 1.0}}),
            (1,    {0: {0: 1.0, 4: 1.0}, 1: {2: 1.0, 5: 1.0}, 2: {1: 1.0, 3: 2.0, 6: 2.0}}),
            (2,    {0: {0: 1.0, 4: 3.0}, 1: {2: 2.0, 5: 1.0}, 2: {1: 1.0, 3: 1.0, 6: 1.0}}),
            # After query 1,2,0: peek shows that token 2 would match at indices 1,3,6 with match len 2,1,1, etc.
            (0,    {0: {0: 1.0, 4: 1.0}, 1: {2: 1.0, 5: 4.0}, 2: {1: 2.0, 3: 1.0, 6: 1.0}}),
            (2,    {0: {0: 1.0, 4: 2.0}, 1: {2: 3.0, 5: 1.0}, 2: {1: 1.0, 3: 1.0, 6: 1.0}}),
            ]
        self.run_test(matcher, test)

    def test2(self):
        matcher = IncrementalTokenMatcher(['a', 'b', 'b', 'b'])
        test = [
            (None, {'a': {0: 1.0}, 'b': {1: 1.0, 2: 1.0, 3: 1.0}}),
            ('b',  {'a': {0: 1.0}, 'b': {1: 1.0, 2: 2.0, 3: 2.0}}),
            ('b',  {'a': {0: 1.0}, 'b': {1: 1.0, 2: 2.0, 3: 3.0}}),
            ('b',  {'a': {0: 1.0}, 'b': {1: 1.0, 2: 2.0, 3: 3.0}}),
            ('a',  {'a': {0: 1.0}, 'b': {1: 2.0, 2: 1.0, 3: 1.0}}),
            ]
        self.run_test(matcher, test)

class TestDecoding(unittest.TestCase):

    def run_test(self, matcher, seq, expected_cumulative_penalty):
        cumulative_penalty = 0
        for tok in seq:
            penalties = matcher.peek()
            print(sorted_dict(penalties))
            penalty = matcher.increment_query(tok)
            print('penalty={}'.format(penalty))
            # assert(penalty == max(penalties[tok].values())), 'expected: {}, actual: {}'.format(max(penalties[tok].values()), penalty)
            cumulative_penalty += penalty
        self.assertAlmostEqual(cumulative_penalty, expected_cumulative_penalty)

    def test1(self):
        matcher = ExtractiveLengthPenalty(['a', 'b', 'c', 'b', 'c', 'd'],
                                          penalty_fct=LengthFunction.create('linear'))
        seq = ['b', 'c', 'd']
        self.run_test(matcher, seq, 3.0)

    def test1b(self):
        matcher = ExtractiveLengthPenalty(['a', 'b', 'c', 'b', 'c', 'd'],
                                          penalty_fct=LengthFunction.create('linear'))
        seq = ['b', 'c', 'x']
        self.run_test(matcher, seq, 2.0)

    def test2(self):
        matcher = ExtractiveLengthPenalty(['a', 'b', 'c', 'b', 'c', 'd'],
                                          penalty_fct=LengthFunction.create('log_exp', (2,1)))
        seq = ['b', 'c', 'd']
        self.run_test(matcher, seq, 9.0)

    def test3(self):
        matcher = ExtractiveLengthPenalty(['a', 'b', 'c', 'b', 'c', 'd'],
                                          penalty_fct=LengthFunction.create('log_exp', (2,1)))
        seq = ['b', 'c', 'd', 'a']
        # The penalty should be 3**2 + 1**2 = 10.0
        self.run_test(matcher, seq, 10.0)

    def test3_neg(self):
        matcher = ExtractiveLengthPenalty(['a', 'b', 'c', 'b', 'c', 'd'],
                                          penalty_fct=LengthFunction.create('neg_log_exp', (2,1)))
        seq = ['b', 'c', 'd', 'a']
        # The penalty should be -3**2 + -1**2 = -10.0 (it's en extraction reward)
        self.run_test(matcher, seq, -10.0)

    def test_weights1(self):
        def weights_fct(token):
            w = {'a': 0.4, 'b': 0.5, 'c': 0.5, 'd': 0.6}
            return w[token]
        matcher = ExtractiveLengthPenalty(['a', 'b', 'c', 'b', 'c', 'd'],
                                          token_len_fct=weights_fct,
                                          penalty_fct=LengthFunction.create('log_exp', (2,1)))
        seq = ['b', 'c', 'd', 'a']
        # The penalty should be (.5+.5+.6)**2 + .5**2 = 2.72
        self.run_test(matcher, seq, 2.72)

    def test_increment(self):
        matcher = ExtractiveLengthPenalty(['a', 'b', 'c', 'b', 'c', 'd'])
        matcher.increment_query('b')
        matcher.increment_query('c')
        self.assertEqual(matcher.query, ['b', 'c'])

if __name__ == '__main__':
    unittest.main()

