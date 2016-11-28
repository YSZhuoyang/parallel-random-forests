
#ifndef _BASIC_DATA_STRUCTURES_H_
#define _BASIC_DATA_STRUCTURES_H_

#include <vector>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/access.hpp>

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

        // Serialization
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(
            Archive &ar,
            const unsigned int version )
        {
            ar & gini;
            ar & giniSplit;
            ar & featureIndex;
            ar & threshold;
            ar & classIndex;
            ar & childrenVec;
        }
    };
}

#endif
