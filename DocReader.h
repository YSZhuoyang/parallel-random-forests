
#include "BaseDataStructures.h"
#include <stdlib.h>
#include <cstdio>


using namespace BasicDataStructures;
using namespace std;

class DocReader
{
#define READ_LINE_MAX 4000
#define TOKEN_LENGTH_MAX 20
#define KEYWORD_ATTRIBUTE "@ATTRIBUTE"
#define KEYWORD_DATA "@DATA"
#define KEYWORD_NUMERIC "NUMERIC"

public:
    DocReader();
    ~DocReader();

    void Read( const char* fileName );
    void GetClassAttr( vector<char*>& cv );
    void GetAttr( vector<char*>& fv );
    void GetItems( vector<Item*>& iv );


private:
    bool StrEqual( const char str1[], const char str2[] );

    vector<char*> classVec;
    vector<char*> featureVec;
    vector<Item*> itemVec;

    //ReadingState state;
};
