
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
        const vector<Instance>& iv );
    void BuildTree( const unsigned int numFeaToSelect );
    void PrintTree( const TreeNode* iter, unsigned int h );
    void DestroyNode( TreeNode* node );
    TreeNode* GetRoot();

private:
    TreeNode* Split(
        const vector<unsigned int>& iiv,
        unsigned int* featureIndexArray,
        unsigned int height );
    double ComputeGini( const vector<unsigned int>& iiv );
    double ComputeEntropy( const vector<unsigned int>& iiv );
    // Count instances of each class
    unsigned int* GetDistribution(
        const vector<unsigned int>& iiv );
    void LabelNode(
        TreeNode* node,
        const vector<unsigned int>& iiv );

    vector<char*> classVec;
    vector<NumericAttr> featureVec;
    vector<Instance> instanceVec;

    unsigned int numFeaturesToSelect;
    unsigned int numFeaturesTotal;
    unsigned short numClasses;
    
    TreeNode* root = nullptr;
};

#endif
