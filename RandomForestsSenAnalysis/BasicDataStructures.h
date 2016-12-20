
#ifndef _BASIC_DATA_STRUCTURES_H_
#define _BASIC_DATA_STRUCTURES_H_

#include <vector>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/access.hpp>


using namespace std;

namespace BasicDataStructures
{
    struct Instance
    {
        double* featureAttrArray;
        unsigned short classIndex;
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
        // double gini;
        // double giniSplit;
        double threshold;
        unsigned int featureIndex;
        unsigned short classIndex;
        bool labeled;
        vector<TreeNode*> childrenVec;
        
        // Serialization
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(
            Archive &ar,
            const unsigned int version )
        {
            // ar & gini;
            // ar & giniSplit;
            ar & threshold;
            ar & featureIndex;
            ar & classIndex;
            ar & labeled;
            ar & childrenVec;
        }
    };
}

#endif
