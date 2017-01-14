
#ifndef _TREE_BUILDER_H_
#define _TREE_BUILDER_H_

#include "BasicDataStructures.h"
#include "Helper.h"

#include <stdio.h>
#include <string.h>


using namespace BasicDataStructures;
using namespace MyHelper;

class TreeBuilder
{
#define MIN_NODE_SIZE          1
#define MIN_NODE_SIZE_TO_SPLIT 2
#define NUM_CHILDREN           2

public:
    TreeBuilder();
    ~TreeBuilder();

    void Init(
        const vector<NumericAttr>& fv,
        const vector<char*>& cv,
        const Instance* it,
        const unsigned int numInstances );
    void BuildTree( const unsigned int numFeaToSelect );
    void PrintTree( const TreeNode* iter, unsigned int h );
    void DestroyNode( TreeNode* node );
    TreeNode* GetRoot();

private:
    TreeNode* Split(
        ValueIndexPair* valueIndexPairArr,
        unsigned int* featureIndexArray,
        const unsigned int* parentClassDist,
        const unsigned int numInstances,
        unsigned int height );
    inline double ComputeGini(
        const unsigned int* classDistribution,
        const unsigned int numInstances );
    inline double ComputeEntropy(
        const unsigned int* classDistribution,
        const unsigned int numInstances );
    inline void LabelNode(
        TreeNode* node,
        const unsigned int* classDistribution );

    vector<char*> classVec;
    vector<NumericAttr> featureVec;
    const Instance* instanceTable;

    unsigned int numFeaturesToSelect;
    unsigned int numFeaturesTotal;
    unsigned int numInstancesTotal;
    unsigned short numClasses;
    
    TreeNode* root = nullptr;
};

#endif
