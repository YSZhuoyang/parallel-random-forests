
#ifndef _CLASSIFIER_H_
#define _CLASSIFIER_H_

#include "TreeBuilder.h"

#include <mpi.h>


class Classifier
{
#define MPI_ROOT_ID 0
#define NUM_FEATURES_PER_TREE 10

public:
    Classifier(
    const vector<NumericAttr>& fv, 
    const vector<char*>& cv );
    ~Classifier();

    void Train( const vector<Item>& iv );
    void Classify( const vector<Item>& iv );


private:
    // Return the index of the predicted class
    int Classify(
        const Item& item, 
        const unsigned int numClasses );

    vector<char*> classVec;
    vector<NumericAttr> featureVec;
    
    TreeBuilder treeBuilder;
    vector<TreeNode*> rootVec;

    // MPI status
    int mpiNodeId;
    int numMpiNodes;
    int mpiInitialized;
};

#endif
