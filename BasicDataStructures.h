
#ifndef _BASIC_DATA_STRUCTURES_H_
#define _BASIC_DATA_STRUCTURES_H_

#include <vector>


using namespace std;

namespace BasicDataStructures
{
    struct Item
    {
        int* featureAttrArray;
        unsigned short classIndex;
    };

    struct NumericAttr
    {
        char* name;
        int min;
        int max;
        int mean;                // Not used.
        float bucketSize;        // Not used.
        unsigned int numBuckets; // Not used.
    };

    struct TreeNode
    {
        float gini;
        float giniSplit;
        unsigned int featureIndex;
        int threshold;
        int classIndex;
        vector<TreeNode*> childrenVec;
    };
}

#endif
