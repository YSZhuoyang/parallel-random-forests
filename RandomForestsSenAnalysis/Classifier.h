
#ifndef _CLASSIFIER_H_
#define _CLASSIFIER_H_

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <fstream>
#include "TreeBuilder.h"

class Classifier
{
//#define NUM_TREES             5
//#define NUM_FEATURES_PER_TREE 10
#define MODEL_FILE_PATH       "./RandomForestsSenAnalysis/model.serial"

public:
    Classifier();
    ~Classifier();

    void Configure(
        unsigned int numTrees,
        unsigned int numFeaPerTree);
    void Train(
        const vector<Item>& iv, 
        const vector<NumericAttr>& fv, 
        const vector<char*>& cv );
    float Classify(
        const vector<Item>& iv, 
        const vector<char*>& cv );


private:
    // Return the index of the predicted class
    int Classify( const Item& item );
    void SaveModel();
    void LoadModel();

    vector<char*> classVec;
    vector<NumericAttr> featureVec;
    
    TreeBuilder treeBuilder;
    vector<TreeNode*> rootVec;

    // Settings
    unsigned int numTrees      = 1;
    unsigned int numFeaPerTree = 1;
};

#endif
