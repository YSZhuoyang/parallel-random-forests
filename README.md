# Parallelized Random Forests
A parallelized version of random forests learning algorithm.
* Refer to weka random tree construction and C4.5 initial tree construction.
* Support Infogain and Gini index used as split criteria.
* No need of pruning.
* Try it out (post a get request to the below address with two attributes: 'num_trees' and 'num_features_per_tree'):

        http://my-apis-env.ap-southeast-2.elasticbeanstalk.com/rf_train
        e.g.
        http://my-apis-env.ap-southeast-2.elasticbeanstalk.com/rf_train?num_trees=10&num_features_per_tree=6

### Configuration
* In 'Classifier.h', change following variables:

        NUM_TREES             // Number of trees to construct
        NUM_FEATURES_PER_TREE // Number of features to be considered in each node for finding best split criteria

* In 'TreeBuilder.h', change following variables:

        MIN_NODE_SIZE          // Minimum size of a node that can be considered as a leaf
        MIN_NODE_SIZE_TO_SPLIT // Minimum size of a node that can be further split

### Parallelization
* Multiple openmpi processes collaboration.
* Multi-threading within single mpi process with openmp.

### Dataset and testing
* Sentiment analysis of 50000 movie reviews from IMDb (25000 for training, 25000 for testing).
* Used top 1000 words with highest frequencies of occurrences, achieved 83% accuracy without attributes selection.
* Testing environment: vlsci clusters
* Linear scalable speedup.

## Terms of use for the dataset

    Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011).
    Learning Word Vectors for Sentiment Analysis.
    The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).
