
#ifndef _TREE_BUILDER_H_
#define _TREE_BUILDER_H_

#include "BasicDataStructures.h"
#include "Helper.h"

#include <stdio.h>
#include <string.h>
#include <math.h>


using namespace BasicDataStructures;
using namespace MyHelper;

class TreeBuilder
{
#define MIN_NODE_SIZE          1
#define MIN_NODE_SIZE_TO_SPLIT 1
#define NUM_CHILDREN           2

public:
    TreeBuilder();
    ~TreeBuilder();

    void Init(
        const vector<NumericAttr>& fv,
        const vector<char*>& cv,
        const unsigned int nfToSelect );
    void BuildTree( const vector<Item>& iv );
    void PrintTree( const TreeNode* iter, unsigned int h );
    void DestroyNode( TreeNode* node );
    TreeNode* GetRoot();

private:
    TreeNode* Split(
        const vector<Item>& iv,
        unsigned int* featureIndexArray,
        unsigned int height );
    float ComputeGini( const vector<Item>& iv );
    float ComputeEntropy( const vector<Item>& iv );
    // Count items of each class
    unsigned int* GetDistribution( const vector<Item>& iv );
    void LabelNode( TreeNode* node, const vector<Item>& iv );

    vector<char*> classVec;
    vector<NumericAttr> featureVec;

    unsigned int numFeaturesToSelect;
    unsigned int numFeaturesTotal;
    unsigned short numClasses;
    
    TreeNode* root = nullptr;
};

#endif
