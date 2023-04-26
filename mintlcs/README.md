# mintlcs

This Python module **mintlcs** implements the [longest common subsequence](https://en.wikipedia.org/wiki/Longest_common_subsequence_problem) (LCS) algorithm in C++.
  
The [longest common subsequence problem](https://en.wikipedia.org/wiki/Longest_common_subsequence_problem) is the problem of finding the longest subsequence in two strings, i.e., a substring with gaps allowed.

This C++ implementation is about 60x faster than a vanilla Python implementation.

Install
-------

To install, type `pip install ./mintlcs`.

If a C++11 support error appears, install C++ first:

    sudo yum install gcc-c++


Example
-------

    $ python
    > import mintlcs
    > s1 = 'He ate a burrito for breakfast'
    > s2 = 'He had a burrito'
    > mintlcs.lcs(s1.split(), s2.split())
    3

It is `3` in the example because the longest common subsequence is "He ... a burrito".
