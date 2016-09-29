
#include "TreeBuilder.h"


TreeBuilder::TreeBuilder()
{
    
}

TreeBuilder::~TreeBuilder()
{
    
}

void TreeBuilder::Init(
    const vector<NumericAttr>& fv, 
    const vector<char*>& cv, 
    const unsigned int nf )
{
    featureVec = fv;
    classVec = cv;
    numFeatures = nf;
    numClasses = classVec.size();

    giniSplitThreshold = 0.002f;
}

void TreeBuilder::BuildTree(
    const vector<Item>& iv, 
    unsigned int* featureIndexArr )
{
    unsigned int featureIndexArraySize = 
        numFeatures * sizeof( unsigned int );

    root = Split( iv, featureIndexArr, featureIndexArraySize, 0 );
}

TreeNode* TreeBuilder::Split(
    const vector<Item>& iv, 
    unsigned int* featureIndexArray, 
    const unsigned int featureIndexArraySize, 
    unsigned int height )
{
    float giniSplitMax = 0.0f;

    // No item exists in this group, return empty pointer.
    if (iv.empty())
    {
        free( featureIndexArray );
        featureIndexArray = nullptr;

        return nullptr;
    }

    // Compute gini of this node
    float giniParent = ComputeGini( iv );

    if (iv.size() < MIN_NUM_ITEMS_PER_NODE || giniParent < 0.001f)
    {
        TreeNode* leaf = new TreeNode;
        LabelNode( leaf, iv );
        
        return leaf;
    }

    unsigned int itemSize = iv.size();
    unsigned int selectedIndex = numFeatures;
    int selectedThreshold;
    vector<vector<Item>> selectedChildren;

    // Find split feature and threshold
    for (unsigned int index = 0; index < numFeatures; index++)
    {
        unsigned int i = featureIndexArray[index];

        if (i == numFeatures) continue;

        // Get all values of that feature and sort them.
        int* valueArr = (int*) malloc( itemSize * sizeof( int ) );
        for (unsigned int itemIndex = 0; itemIndex < itemSize; itemIndex++)
            valueArr[itemIndex] = iv[itemIndex].featureAttrArray[i];
        qsort( valueArr, itemSize, sizeof( int ), Compare );

        // Find split threshold
        for (unsigned int itemIndex = 0; itemIndex < itemSize; itemIndex++)
        {
            if (itemIndex > 0 && valueArr[itemIndex] == valueArr[itemIndex - 1])
                continue;
            
            vector<vector<Item>> groups( 2 );

            for (const Item& item : iv)
                groups[item.featureAttrArray[i] > valueArr[itemIndex]].push_back( item );

            float giniSplit = giniParent;
            
            // Compute gini of children
            for (vector<Item>& group : groups)
            {
                float giniChild = ComputeGini( group );
                float numChildren = group.size();
                giniSplit -= numChildren / itemSize * giniChild;
            }

            // Get max gini split and related feature
            if (giniSplitMax < giniSplit)
            {
                giniSplitMax = giniSplit;
                selectedChildren = groups;
                selectedThreshold = valueArr[itemIndex];
                selectedIndex = index;
            }
        }

        free( valueArr );
        valueArr = nullptr;
    }

    //printf( "\n----------------------------------------\n");
    //printf( "Height: %d\n", height );

    // Create parent node
    TreeNode* node = new TreeNode;

    // All features have been used, or gini split exceeds threshold,
    // thus have reached leaf node.
    if (giniSplitMax <= giniSplitThreshold)
    {
        LabelNode( node, iv );

        //printf( "Leaf node labeled with class index: %u\n", node->classIndex );
    }
    // Split node
    else
    {
        unsigned int selectedFeatureIndex = featureIndexArray[selectedIndex];

        //printf( "Feature selected: %s\n", featureVec[selectedFeatureIndex].name );
        //printf( "Gini of parent: %f\n", giniParent );
        //printf( "Max Gini split get: %f\n", giniSplitMax );
        
        node->featureIndex = selectedFeatureIndex;
        node->threshold    = selectedThreshold;
        node->gini         = giniParent;
        node->giniSplit    = giniSplitMax;
        node->classIndex   = -1;

        // Turn off the flag of selected feature
        featureIndexArray[selectedIndex] = numFeatures;
        height++;

        bool nodeLabeled = false;

        // Split children
        for (vector<Item> childGroup : selectedChildren)
        {
            unsigned int* featureIndexArrayCopy = 
                (unsigned int*) malloc( featureIndexArraySize );
            memcpy( featureIndexArrayCopy, featureIndexArray, 
                featureIndexArraySize );

            TreeNode* childNode = Split( childGroup, 
                featureIndexArrayCopy, featureIndexArraySize, height );
            if (childNode == nullptr) nodeLabeled = true;
            node->childrenVec.push_back( childNode );
        }

        if (nodeLabeled) LabelNode( node, iv );
    }

    free( featureIndexArray );
    featureIndexArray = nullptr;
    
    return node;
}

void TreeBuilder::PrintTree( const TreeNode* iter )
{
    if (iter == nullptr || iter->classIndex == -1) return;

    printf( "Feature: %s\n", featureVec[iter->featureIndex].name );

    for (const TreeNode* child : iter->childrenVec)
        PrintTree( child );
}

void TreeBuilder::SetGiniSplitThreshold( float gst )
{
    giniSplitThreshold = gst;
}

TreeNode* TreeBuilder::GetRoot()
{
    return root;
}

float TreeBuilder::ComputeGini( const vector<Item>& iv )
{
    float totalItemCount = iv.size();

    if (totalItemCount == 0) return 0.0f;

    unsigned int* bucketArray = 
        (unsigned int*) calloc( numClasses, sizeof( unsigned int ) );
    for (const Item& item : iv) bucketArray[item.classIndex]++;

    float gini = 1.0f;

    for (unsigned short i = 0; i < numClasses; i++)
    {
        float temp = (float) bucketArray[i] / totalItemCount;
        gini -= temp * temp;
    }

    free( bucketArray );
    bucketArray = nullptr;

    return gini;
}

void TreeBuilder::LabelNode( TreeNode* node, const vector<Item>& iv )
{
    if (node == nullptr) return;

    unsigned int* classCounters = 
        (unsigned int*) calloc( numClasses, sizeof( unsigned int ) );
    for (const Item& item : iv) classCounters[item.classIndex]++;

    // Select the class of the largest class group.
    node->classIndex = getIndexOfMax( classCounters, numClasses );
    free( classCounters );
}

void TreeBuilder::DestroyNode( TreeNode* node )
{
    if (node == nullptr) return;

    if (!node->childrenVec.empty())
    {
        for (TreeNode* child : node->childrenVec) DestroyNode( child );
        node->childrenVec.clear();
    }

    delete node;
    node = nullptr;
}
