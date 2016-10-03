# Parallelized Random Forests
A parallelized version of random forests learning algorithm.

### Decision tree construction
* Refer to C4.5 initial tree construction.
* Gini index was used as split criteria.
* No need of pruning.

### Parallelization
* Multiple openmpi processes collaboration.
* Multi-threading within single mpi process with openmp.

### Dataset and testing
* Sentiment analysis of 50000 movie reviews from IMDb (25000 for training, 25000 for testing).
* Used top 1000 words with highest frequencies of occurrences, achieved 77% accuracy without attributes selection and data cleanning.

## Terms of use for the dataset

    Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011).
    Learning Word Vectors for Sentiment Analysis.
    The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).
