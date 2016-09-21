
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
        unsigned int bucketSize;
        unsigned int numBuckets;
    };

    struct TreeNode
    {
        float gini;
        float giniSplit;
        char* feature;
        vector<TreeNode*> childrenVec;
    };
}
