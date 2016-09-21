
#include "BasicDataStructures.h"


using namespace BasicDataStructures;

class TreeBuilder
{
public:
    TreeBuilder();
    ~TreeBuilder();

    void BuildTree( vector<Item>& iv, vector<NumericAttr> fv, vector<char*> cv );
    TreeNode* GetRoot();

private:
    void DestroyNode( TreeNode* node );

    TreeNode* root;
    vector<char*> classVec;
    vector<NumericAttr> featureVec;
    vector<Item> itemVec;
};






