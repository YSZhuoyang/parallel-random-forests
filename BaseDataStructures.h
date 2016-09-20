
#include <vector>


using namespace std;

namespace BasicDataStructures
{
    //enum class ReadingState { READING_FEATURE_NAMES, READING_FEATURES, DOING_NOTHING };

    struct Item
    {
        unsigned int* featureArr;
        unsigned short classIndex;
    };

    struct TreeNode
    {
        float gini;
        float giniSplit;
        char* feature;
        vector<TreeNode*> childrenVec;
    };

    class DecisionTree
    {
    public:
        DecisionTree()
        {

        }

        ~DecisionTree()
        {
            DestroyNode( root );
        }

        void BuildTree( vector<Item*>& iv )
        {
            
        }

    private:
        void DestroyNode( TreeNode* node )
        {
            if (!node->childrenVec.empty())
            {
                for (TreeNode* child : node->childrenVec) DestroyNode( child );
                node->childrenVec.clear();
            }

            delete node;
            node = nullptr;
        }

        TreeNode* root;
    };
}
