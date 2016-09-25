
#ifndef _CLASSIFIER_H_
#define _CLASSIFIER_H_

#include "TreeBuilder.h"

class Classifier
{
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

    //vector<TreeBuilder> treeBuilderVec;
    //vector<TreeNode*> rootVec;
    
    TreeBuilder treeBuilder;
    TreeNode* root = nullptr;
};

#endif
