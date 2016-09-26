
#include "TreeBuilder.h"


TreeBuilder::TreeBuilder()
{
    giniSplitThreshold = 0.0018f;
}

TreeBuilder::~TreeBuilder()
{
    DestroyNode( root );
}

void TreeBuilder::BuildTree(
    const vector<Item>& iv, 
    const vector<NumericAttr>& fv, 
    const vector<char*>& cv )
{
    featureVec = fv;
    classVec = cv;

    unsigned int numFeatures = featureVec.size();
    // Initialize a feature index array indicating feature has not been
    // used in the parent nodes.
    unsigned short* featureIndexArray = 
        (unsigned short*) malloc( numFeatures * sizeof( unsigned short ) );
    for (unsigned int i = 0; i < numFeatures; i++) featureIndexArray[i] = 1;
    size_t featureIndexArraySize = numFeatures * sizeof( unsigned short );

    root = Split( iv, featureIndexArray, featureIndexArraySize, 0 );
}

TreeNode* TreeBuilder::Split(
    const vector<Item>& iv, 
    unsigned short* featureIndexArray, 
    unsigned int featureIndexArraySize, 
    unsigned int height )
{
    unsigned int numFeatures = featureVec.size();
    float giniSplitMax = 0.0f;

    // No item exists in this group, return empty pointer.
    if (iv.empty())
    {
        free( featureIndexArray );
        featureIndexArray = nullptr;

        return nullptr;
    }

    // Compute gini of parent
    float giniParent = ComputeGini( iv );

    vector<vector<Item>> selectedChildren;
    unsigned int selectedFeatureIndex = numFeatures;

    for (unsigned int i = 0; i < numFeatures; i++)
    {
        if (!featureIndexArray[i]) continue;

        vector<vector<Item>> groups( featureVec[i].numBuckets );

        // If there are only 2 buckets, then classify them into:
        // a group of 0, and a group of greater than 0.
        if (featureVec[i].numBuckets == 2)
        {
            for (const Item& item : iv)
            {
                if (item.featureAttrArray[i] <= 0)
                {
                    groups[0].push_back( item );
                }
                else
                {
                    groups[1].push_back( item );
                }
            }
        }
        else
        {
            for (const Item& item : iv)
            {
                unsigned int bucketIndex = 
                    (float) (item.featureAttrArray[i] - featureVec[i].min) / 
                    featureVec[i].bucketSize;
                
                if (bucketIndex >= featureVec[i].numBuckets)
                    bucketIndex = featureVec[i].numBuckets - 1;

                groups[bucketIndex].push_back( item );
            }
        }

        float giniSplit = giniParent;
        
        // Compute gini of children
        for (vector<Item>& group : groups)
        {
            float giniChild = ComputeGini( group );
            float numTotal = iv.size();
            float numChildren = group.size();
            giniSplit -= numChildren / numTotal * giniChild;
        }

        // Get max gini split and related feature
        if (giniSplitMax < giniSplit)
        {
            giniSplitMax = giniSplit;
            selectedChildren = groups;
            selectedFeatureIndex = i;
        }
    }

    printf( "\n|--------------------------------------|\n");
    printf( "Height: %d\n", height );

    // Create parent node
    TreeNode* node = new TreeNode;

    // All features have been used, or gini split exceeds threshold,
    // thus have reached leaf node.
    if (giniSplitMax <= giniSplitThreshold)
    {
        LabelNode( node, iv );

        printf( "Leaf node labeled with class index: %u\n", node->classIndex );
    }
    // Split node
    else
    {
        printf( "Feature selected: %s\n", featureVec[selectedFeatureIndex].name );
        //printf( "Gini of parent: %f\n", giniParent );
        printf( "Max Gini split get: %f\n", giniSplitMax );
        
        node->featureIndex = selectedFeatureIndex;
        node->gini         = giniParent;
        node->giniSplit    = giniSplitMax;
        node->classIndex   = -1;

        // Turn off the flag of selected feature
        featureIndexArray[selectedFeatureIndex] = 0;
        height++;

        bool nodeLabeled = false;

        // Split children
        for (vector<Item> childGroup : selectedChildren)
        {
            unsigned short* featureIndexArrayCopy = 
                (unsigned short*) malloc( featureIndexArraySize );
            memcpy( featureIndexArrayCopy,
                featureIndexArray, 
                featureIndexArraySize );

            TreeNode* childNode = Split( childGroup, 
                featureIndexArrayCopy, featureIndexArraySize, height );
            if (childNode != nullptr) nodeLabeled = true;
            node->childrenVec.push_back( childNode );
        }

        if (nodeLabeled) LabelNode( node, iv );
    }

    free( featureIndexArray );
    featureIndexArray = nullptr;
    
    return node;
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
    unsigned short numClasses = classVec.size();
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

    unsigned int* classCounters = (unsigned int*) calloc( classVec.size(), sizeof( unsigned int ) );
    for (const Item& item : iv) classCounters[item.classIndex]++;

    // Select the class of the largest class group.
    node->classIndex = getIndexOfMax( classCounters, classVec.size() );
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
