
#ifndef _CLASSIFIER_H_
#define _CLASSIFIER_H_

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <fstream>
//#include <omp.h>
#include "TreeBuilder.h"

class Classifier
{
#define MODEL_FILE_PATH       "./RandomForestsSenAnalysis/model.serial"

public:
    Classifier();
    ~Classifier();

    void Configure(
        unsigned int numTrees,
        unsigned int numFeaPerTree );
    void Train(
        const vector<Item>& iv, 
        const vector<NumericAttr>& fv, 
        const vector<char*>& cv );
    char* Analyze(
        const char* str,
        const vector<NumericAttr>& featureVec,
        const vector<char*>& cv );
    float Test(
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
    unsigned int numFeaPerTree = 4;
};

#endif
