# Parallel Random Forests
A parallelized version of random forests learning algorithm.
* Based on Weka's implementation of Breiman random forest construction.
* Support continuous features, which are repeatedly used during split.
* Support using Infogain / Gini impurity as split criteria.
* 2x speedup over Weka Random Forests (for high dimensional dataset).
* Scalable speedup by OpenMP and Open mpi parallelization.

<p align="center">
<img src="https://github.com/YSZhuoyang/Parallelized-Random-Forests/blob/rf/serial/Comp/speed.PNG" alt="Comparison" width= "430px" height="320px" />
<img src="https://github.com/YSZhuoyang/Parallelized-Random-Forests/blob/rf/serial/Comp/acc.PNG" alt="Comparison" width= "430px" height="330px" />
</p>

### Configuration
* In 'Classifier.h', change following variables:

        NUM_TREES               // Number of trees to construct
        RANDOM_FEATURE_SET_SIZE // Number of random features to be considered for finding the best split candidates

* In 'TreeBuilder.h', change following variables:

        MIN_NODE_SIZE           // Minimum size of a node that can be considered as a leaf
        MIN_NODE_SIZE_TO_SPLIT  // Minimum size of a node that can be further split

### Dataset and testing
* Sentiment analysis of 50000 movie reviews from IMDb (25000 for training, 25000 for testing).
* Used top 10/50/200/1000 words with highest frequencies of occurrences, achieved the same accuracies.
* Test environment: Ubuntu Gnome 16.04, vlsci clusters (for distributed execution on clusters)

## Terms of use for the dataset

    Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011).
    Learning Word Vectors for Sentiment Analysis.
    The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).
