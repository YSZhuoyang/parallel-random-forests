
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
    //float giniSplitMax = 0.0f;
    float infoGainMax = 0.0f;
    unsigned int numItems = iv.size();

    // The node is too small thus it is ignored.
    if (numItems < MIN_NODE_SIZE)
    {
        free( featureIndexArray );
        featureIndexArray = nullptr;

        return nullptr;
    }
    // The node is small, make it a leaf node.
    else if (numItems < MIN_NODE_SIZE_TO_SPLIT)
    {
        TreeNode* leaf = new TreeNode;
        LabelNode( leaf, iv );
        
        return leaf;
    }

    //float giniParent = ComputeGini( iv );
    // Compute entropy of this node.
    float entropyParent = ComputeEntropy( iv );

    // All items are in the same classes.
    if (entropyParent == 0.0f)
    {
        TreeNode* leaf = new TreeNode;
        LabelNode( leaf, iv );
        
        return leaf;
    }

    unsigned int selectedIndex = numFeatures;
    int selectedThreshold;
    vector<vector<Item>> selectedChildren;

    // Find the best split feature and threshold
    for (unsigned int index = 0; index < numFeatures; index++)
    {
        unsigned int i = featureIndexArray[index];

        if (i == numFeatures) continue;

        // Get all values of that feature and sort them.
        int* valueArr = (int*) malloc( numItems * sizeof( int ) );
        for (unsigned int itemIndex = 0; itemIndex < numItems; itemIndex++)
            valueArr[itemIndex] = iv[itemIndex].featureAttrArray[i];
        qsort( valueArr, numItems, sizeof( int ), Compare );
        unsigned int numUniqueEle = removeDuplicates( valueArr, numItems );

        // Find split threshold
        for (unsigned int valueIndex = 0; valueIndex < numUniqueEle; valueIndex++)
        {
            vector<vector<Item>> groups( 2 );

            for (const Item& item : iv)
                groups[item.featureAttrArray[i] > valueArr[valueIndex]].push_back( item );

            //float giniSplit = giniParent;
            float infoGain = entropyParent;
            
            // Compute entropy of children
            for (vector<Item>& group : groups)
            {
                //float giniChild = ComputeGini( group );
                float entropyChild = ComputeEntropy( group );
                float numChildren = group.size();
                //giniSplit -= (float) numChildren / (float) numItems * giniChild;
                infoGain -= (float) numChildren / (float) numItems * entropyChild;
            }

            // Get max info gain and related feature
            //if (giniSplitMax < giniSplit)
            if (infoGainMax < infoGain )
            {
                //giniSplitMax = giniSplit;
                infoGainMax = infoGain;
                selectedChildren = groups;
                selectedThreshold = valueArr[valueIndex];
                selectedIndex = index;
            }
        }

        free( valueArr );
        valueArr = nullptr;
    }

    //printf( "\n----------------------------------------\n");
    //printf( "Height: %d\n", height );

    TreeNode* node = new TreeNode;

    // All features have been used, 
    // or gini split / info gain exceeds threshold,
    // thus have reached leaf node.
    if (selectedIndex == numFeatures)
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
        //node->gini         = giniParent;
        //node->giniSplit    = giniSplitMax;
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
    if (iter == nullptr || iter->classIndex != -1) return;

    printf( "Feature: %s\n", featureVec[iter->featureIndex].name );

    for (const TreeNode* child : iter->childrenVec)
        PrintTree( child );
}

TreeNode* TreeBuilder::GetRoot()
{
    return root;
}

float TreeBuilder::ComputeEntropy( const vector<Item>& iv )
{
    float totalItemCount = iv.size();
    if (totalItemCount == 0) return 0.0f;

    unsigned int* bucketArray = GetDistribution( iv );
    float entropy = 0.0f;

    for (unsigned short i = 0; i < numClasses; i++)
    {
        float temp = (float) bucketArray[i] / totalItemCount;
        entropy -= temp * log2f( temp );
    }
    
    free( bucketArray );
    bucketArray = nullptr;

    return entropy;
}

float TreeBuilder::ComputeGini( const vector<Item>& iv )
{
    float totalItemCount = iv.size();
    if (totalItemCount == 0) return 0.0f;

    unsigned int* bucketArray = GetDistribution( iv );
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

unsigned int* TreeBuilder::GetDistribution( const vector<Item>& iv )
{
    unsigned int* bucketArray = 
        (unsigned int*) calloc( numClasses, sizeof( unsigned int ) );
    for (const Item& item : iv) bucketArray[item.classIndex]++;

    return bucketArray;
}

void TreeBuilder::LabelNode( TreeNode* node, const vector<Item>& iv )
{
    if (node == nullptr) return;

    unsigned int* bucketArray = GetDistribution( iv );
    // Select the class of the largest class group.
    node->classIndex = getIndexOfMax( bucketArray, numClasses );

    free( bucketArray );
    bucketArray = nullptr;
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
