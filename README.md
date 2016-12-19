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

## Build and deploy locally
* Install boost on Ubuntu: run 'sudo apt install libboost-serialization-dev'
* Under each module directories, run 'make' to compile C shared lib files

## Deploy on aws elastic beanstalk
* Create a folder named '.ebextensions' under root dir
* Cd to '.ebextensions', create a config file named '.config'
* Add following to '.config' file:


        container_commands:
          01-command:
            command: "cd $(find . -name RandomForestsSenAnalysis) && sudo make"

        packages:
          yum:
            gcc-c++: []
            boost-devel: []

## Author
* Oscar Yu

## Terms of use for the dataset

    Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011).
    Learning Word Vectors for Sentiment Analysis.
    The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).
