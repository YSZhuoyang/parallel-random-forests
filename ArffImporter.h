
#ifndef _ARFF_IMPORTER_H_
#define _ARFF_IMPORTER_H_


#include "BasicDataStructures.h"
#include "Helper.h"

#include <stdio.h>
#include <string.h>


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
    Instance* BuildInstTable();
    TransInstTable BuildTransposedInstTable();
    unsigned int GetNumInstances();


private:
    vector<char*> classVec;
    vector<NumericAttr> featureVec;
    vector<Instance> instanceVec;

    TransInstTable transInstTable;
    Instance* instanceTable   = nullptr;
    double* instanceBuff      = nullptr;

    unsigned int numFeatures  = 0;
    unsigned int numInstances = 0;
    unsigned short numClasses = 0;
};

#endif
