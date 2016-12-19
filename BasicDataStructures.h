
#ifndef _BASIC_DATA_STRUCTURES_H_
#define _BASIC_DATA_STRUCTURES_H_

#include <vector>


using namespace std;

namespace BasicDataStructures
{
    struct Instance
    {
        float* featureAttrArray;
        unsigned short classIndex;
    };

    struct NumericAttr
    {
        char* name;
        float min;
        float max;
        float mean;              // Not used.
        float bucketSize;        // Not used.
        unsigned int numBuckets; // Not used.
    };

    struct TreeNode
    {
        //float gini;
        //float giniSplit;
        float threshold;
        bool labeled;
        unsigned int featureIndex;
        unsigned short classIndex;
        vector<TreeNode*> childrenVec;
    };
}

#endif
