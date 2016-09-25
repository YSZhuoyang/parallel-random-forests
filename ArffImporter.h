
#ifndef _ARFF_IMPORTER_H_
#define _ARFF_IMPORTER_H_


#include "BasicDataStructures.h"
#include <stdlib.h>
#include <cstdio>


using namespace BasicDataStructures;
using namespace std;

class ArffImporter
{
#define READ_LINE_MAX 4000
#define TOKEN_LENGTH_MAX 20

#define KEYWORD_ATTRIBUTE "@ATTRIBUTE"
#define KEYWORD_DATA "@DATA"
#define KEYWORD_NUMERIC "NUMERIC"

public:
    ArffImporter();
    ~ArffImporter();

    void Read( const char* fileName );
    vector<char*> GetClassAttr();
    vector<NumericAttr> GetFeatures();
    vector<Item> GetItems();


private:
    bool StrEqual( const char str1[], const char str2[] );
    // Include string terminator
    unsigned int GetStrLength( const char* str );

    vector<char*> classVec;
    vector<NumericAttr> featureVec;
    vector<Item> itemVec;

    unsigned int numFeatures       = 0;
    unsigned short numClasses      = 0;
};

#endif
