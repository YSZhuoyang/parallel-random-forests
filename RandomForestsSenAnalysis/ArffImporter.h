
#ifndef _ARFF_IMPORTER_H_
#define _ARFF_IMPORTER_H_


#include "BasicDataStructures.h"
#include "Helper.h"

#include <cstdio>


using namespace BasicDataStructures;
using namespace MyHelper;

class ArffImporter
{
#define READ_LINE_MAX     5000
#define TOKEN_LENGTH_MAX  35

#define KEYWORD_ATTRIBUTE "@ATTRIBUTE"
#define KEYWORD_DATA      "@DATA"
#define KEYWORD_NUMERIC   "NUMERIC"

public:
    ArffImporter();
    ~ArffImporter();

    void Read( const char* fileName );
    vector<char*> GetClassAttr();
    vector<NumericAttr> GetFeatures();
    vector<Instance> GetInstances();


private:
    vector<char*> classVec;
    vector<NumericAttr> featureVec;
    vector<Instance> instanceVec;

    unsigned int numFeatures       = 0;
    unsigned short numClasses      = 0;
};

#endif
