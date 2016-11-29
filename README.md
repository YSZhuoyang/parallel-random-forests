# Parallelized Random Forests
A parallelized version of random forests learning algorithm.
* Boost serialization is used for saving trained model by object serializing.

## Install flask with virtualenv
* Run:
        sudo apt install python3-pip
        sudo pip3 install virtualenv
        virtualenv -p python3 venv
        source venv/bin/activate
        pip3 install flask

## Build components
* Install boost on Ubuntu: run 'sudo apt install libboost-serialization-dev'
* Under each module directories, run 'make' to compile C shared lib files

## Author
* Oscar Yu

## Terms of use for the dataset

    Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011).
    Learning Word Vectors for Sentiment Analysis.
    The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).
