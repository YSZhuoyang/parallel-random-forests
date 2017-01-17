# Parallelized Random Forests
A random forest learning algorithm (without bagging).
* Support both Gini impurity and info gain as split criteria.
* Instead of sorting feature value - index pairs, instance data are sorted.
* Testing result showed that sorting instance data is slower than sorting feature value - index pairs.

### Configuration
* In 'Classifier.h', change following variables:

        NUM_TREES             // Number of trees to construct
        NUM_FEATURES_PER_TREE // Number of features to be considered in each node for finding best split criteria

* In 'TreeBuilder.h', change following variables:

        MIN_NODE_SIZE          // Minimum size of a node that can be considered as a leaf
        MIN_NODE_SIZE_TO_SPLIT // Minimum size of a node that can be further split

### Dataset and testing
* Sentiment analysis of 50000 movie reviews from IMDb (25000 for training, 25000 for testing).
* Used top 1000 words with highest frequencies of occurrences, 83% accuracy obtained without attributes selection.

### Author
* Oscar Yu

### Terms of use for the dataset

    Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011).
    Learning Word Vectors for Sentiment Analysis.
    The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).
