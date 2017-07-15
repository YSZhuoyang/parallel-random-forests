# Parallelized Random Forests
A random forest learning algorithm (without bagging).
* Support both Gini impurity and info gain as split criteria.

### Configuration
* In 'Classifier.h', change following variables:

        NUM_TREES               // Number of trees to construct
        RANDOM_FEATURE_SET_SIZE // Number of features to be considered in each node for finding best split criteria

* In 'TreeBuilder.h', change following variables:

        MIN_NODE_SIZE           // Minimum size of a node that can be considered as a leaf
        MIN_NODE_SIZE_TO_SPLIT  // Minimum size of a node that can be further split

### Dataset and testing
* Sentiment analysis of 50000 movie reviews from IMDb (25000 for training, 25000 for testing).
* Used top 1000 words with highest frequencies of occurrences, 40%-50% faster than Weka RF while achieving the same accuracies.

### Author
* Oscar Yu

### Terms of use for the dataset

    Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011).
    Learning Word Vectors for Sentiment Analysis.
    The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).
