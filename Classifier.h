
#ifndef _CLASSIFIER_H_
#define _CLASSIFIER_H_

#include "TreeBuilder.h"

#include <mpi.h>


class Classifier
{
#define MPI_ROOT_ID 0
#define NUM_FEATURES_PER_TREE 10

public:
    Classifier();
    ~Classifier();

    void Train(
        const vector<Item>& iv, 
        const vector<NumericAttr>& fv, 
        const vector<char*>& cv );
    void Classify( const vector<Item>& iv );


private:
    // Return the index of the predicted class
    int Classify( const Item& item );

    vector<char*> classVec;
    vector<NumericAttr> featureVec;
    
    TreeBuilder treeBuilder;
    vector<TreeNode*> rootVec;

    // MPI status
    int mpiInitialized;
};

#endif
