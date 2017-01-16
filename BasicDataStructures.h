
#ifndef _BASIC_DATA_STRUCTURES_H_
#define _BASIC_DATA_STRUCTURES_H_

#include <vector>


using namespace std;

namespace BasicDataStructures
{
    // struct ValueIndexPair
    // {
    //     double featureValue;
    //     unsigned int featureIndex;
    // };

    struct Comp {
        Comp( unsigned int featureId )
        {
            this->featureId = featureId;
        }

        bool operator() ( double* i, double* j )
        {
            return i[featureId] < j[featureId];
        }

        unsigned int featureId;
    };

    struct NumericAttr
    {
        char* name;
        double min;
        double max;
        double mean;             // Not used.
        unsigned int numBuckets; // Not used.
    };

    struct TreeNode
    {
        //double gini;
        //double giniSplit;
        double threshold;
        bool labeled;
        unsigned int featureIndex;
        unsigned short classIndex;
        vector<TreeNode*> childrenVec;
    };
}

#endif
