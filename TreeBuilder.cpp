
#include "TreeBuilder.h"



TreeBuilder::TreeBuilder()
{

}

TreeBuilder::~TreeBuilder()
{
    DestroyNode( root );
}

void TreeBuilder::BuildTree( vector<Item>& iv, vector<NumericAttr> fv, vector<char*> cv )
{
    
}

TreeNode* TreeBuilder::GetRoot()
{
    return root;
}

void TreeBuilder::DestroyNode( TreeNode* node )
{
    if (!node->childrenVec.empty())
    {
        for (TreeNode* child : node->childrenVec) DestroyNode( child );
        node->childrenVec.clear();
    }

    delete node;
    node = nullptr;
}

