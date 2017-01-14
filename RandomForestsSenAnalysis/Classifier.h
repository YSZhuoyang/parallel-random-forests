
#ifndef _CLASSIFIER_H_
#define _CLASSIFIER_H_

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <fstream>
#include <time.h>
#include <omp.h>
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
        const Instance* instanceTable,
        const vector<NumericAttr>& fv,
        const vector<char*>& cv,
        const unsigned int numInstances );
    char* Analyze(
        const char* str,
        const vector<NumericAttr>& featureVec,
        const vector<char*>& cv );
    double Test(
        const vector<char*>& cv,
        const Instance* instanceTable,
        const unsigned int numInstances );


private:
    // Return the index of the predicted class
    unsigned short Classify( const Instance& instance );
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
