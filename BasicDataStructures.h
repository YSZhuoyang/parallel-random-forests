
#ifndef _BASIC_DATA_STRUCTURES_H_
#define _BASIC_DATA_STRUCTURES_H_

#include <vector>


using namespace std;

namespace BasicDataStructures
{
    struct Item
    {
        unsigned int* featureAttrArray;
        unsigned short classIndex;
    };

    struct NumericAttr
    {
        char* name;
        int min;
        int max;
        float bucketSize;
        unsigned int numBuckets;
    };

    struct TreeNode
    {
        float gini;
        float giniSplit;
        unsigned int featureIndex;
        int classIndex;
        vector<TreeNode*> childrenVec;
    };
}

#endif
