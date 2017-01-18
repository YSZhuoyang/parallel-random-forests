
#ifndef _CLASSIFIER_H_
#define _CLASSIFIER_H_

#include "TreeBuilder.h"
#include <omp.h>
#include <time.h>

class Classifier
{
#define NUM_FEATURES_PER_TREE 10
#define NUM_TREES             100

public:
    Classifier();
    ~Classifier();

    void Train(
        const Instance* instanceTable,
        const vector<NumericAttr>& fv,
        const vector<char*>& cv,
        const unsigned int numInstances );
    void Classify( 
        const Instance* instanceTable,
        const unsigned int numInstances );
    char* Analyze(
        const char* str,
        const vector<NumericAttr>& featureVec,
        const vector<char*>& cv );


private:
    // Return the index of the predicted class
    unsigned short Classify( const Instance& instance );

    vector<char*> classVec;
    vector<NumericAttr> featureVec;
    
    TreeBuilder treeBuilder;
    TreeNode* rootArr = nullptr;
};

#endif
