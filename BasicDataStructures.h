
#ifndef _BASIC_DATA_STRUCTURES_H_
#define _BASIC_DATA_STRUCTURES_H_

#include <vector>


using namespace std;

namespace BasicDataStructures
{
    struct Instance
    {
        double* featureAttrArray;
        unsigned short classIndex;
    };

    // Bound index of each instance with one of its feature value
    // and its class index, to minimize memory access time during
    // tree construction
    struct ValueIndexTuple
    {
        double featureValue;
        unsigned int featureIndex;
        unsigned short classIndex;
    };

    struct NumericAttr
    {
        char* name;
        double min;
        double max;
        double mean; // Not used.
    };

    struct TreeNode
    {
        bool empty;
        double threshold;
        unsigned int featureIndex;
        unsigned short classIndex;
        TreeNode* childrenArr;
    };
}

#endif
