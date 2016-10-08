
#ifndef _TREE_BUILDER_H_
#define _TREE_BUILDER_H_

#include "BasicDataStructures.h"
#include "Helper.h"

#include <stdio.h>
#include <string.h>
#include <omp.h>


using namespace BasicDataStructures;
using namespace MyHelper;

class TreeBuilder
{
#define MIN_NODE_SIZE 1
#define MIN_NODE_SIZE_TO_SPLIT 1000

public:
    TreeBuilder();
    ~TreeBuilder();

    void Init(
        const vector<NumericAttr>& fv, 
        const vector<char*>& cv, 
        const unsigned int nf );
    void BuildTree(
        const vector<Item>& iv, 
        unsigned int* featureIndexArr );
    void PrintTree( const TreeNode* iter );
    void DestroyNode( TreeNode* node );
    TreeNode* GetRoot();

private:
    TreeNode* Split(
        const vector<Item>& iv, 
        unsigned int* featureIndexArray, 
        unsigned int featureIndexArraySize, 
        unsigned int height );
    float ComputeGini( const vector<Item>& iv );
    void LabelNode( TreeNode* node, const vector<Item>& iv );

    vector<char*> classVec;
    vector<NumericAttr> featureVec;

    unsigned int numFeatures;
    unsigned short numClasses;
    
    TreeNode* root = nullptr;
};

#endif
