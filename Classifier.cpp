
#include "Classifier.h"


Classifier::Classifier()
{

}

Classifier::~Classifier()
{
    // Destory trees
    for (TreeNode* root : rootVec) treeBuilder.DestroyNode( root );
    rootVec.clear();
}


void Classifier::Train(
    const vector<Item>& iv, 
    const vector<NumericAttr>& fv, 
    const vector<char*>& cv )
{
    classVec = cv;
    featureVec = fv;

    // Randomly select features and build trees.
    unsigned int numFeatures = fv.size();
    
    // Generate an ordered index container, and disorder it.
    unsigned int* randomIndices = 
        (unsigned int*) malloc( numFeatures * sizeof( unsigned int ) );
    for (unsigned int i = 0; i < numFeatures; i++) randomIndices[i] = i;
    randomizeArray( randomIndices, numFeatures, numFeatures / 2 );

    // Build a number of trees each of which having 10 features.
    // What if numFeatures is 51 ?
    unsigned int numTrees = numFeatures / NUM_FEATURES_PER_TREE;
    rootVec.reserve( numTrees );
    treeBuilder.Init( fv, cv, NUM_FEATURES_PER_TREE );

    for (unsigned int treeIndex = 0; treeIndex < numTrees; treeIndex++)
    {
        unsigned int* featureIndexArr = (unsigned int*) 
            malloc( NUM_FEATURES_PER_TREE * sizeof( unsigned int ) );
        memcpy( featureIndexArr, 
            randomIndices + treeIndex * NUM_FEATURES_PER_TREE, 
            NUM_FEATURES_PER_TREE * sizeof( unsigned int ) );

        // Build one tree
        treeBuilder.BuildTree( iv, featureIndexArr );
        rootVec.push_back( treeBuilder.GetRoot() );
    }

    free( randomIndices );
    randomIndices = nullptr;

    //treeBuilder.BuildTree( iv, fv, cv );
    //root = treeBuilder.GetRoot();
}

void Classifier::Classify( const vector<Item>& iv )
{
    if (classVec.empty())
    {
        printf( "Please train the model first.\n" );
        return;
    }

    unsigned int correctCounter = 0;
    unsigned int totalNumber = iv.size();

    for (const Item& item : iv)
        if (Classify( item ) == item.classIndex) correctCounter++;

    float correctRate = (float) correctCounter / (float) totalNumber;
    float incorrectRate = 1.0f - correctRate;

    printf( "Correct rate: %f\n", correctRate );
    printf( "Incorrect rate: %f\n", incorrectRate );
}

int Classifier::Classify( const Item& item )
{
    TreeNode* node = root;
    if (node == nullptr) return -1;

    while (!node->childrenVec.empty())
    {
        unsigned int i = node->featureIndex;

        // If there are only 2 buckets, then classify them into:
        // one group of 0, and another group of greater than 0.
        if (featureVec[i].numBuckets == 2)
        {
            if (item.featureAttrArray[i] <= 0)
            {
                if (node->childrenVec[0] == nullptr)
                    break;
                else
                    node = node->childrenVec[0];
            }
            else
            {
                if (node->childrenVec[1] == nullptr)
                    break;
                else
                    node = node->childrenVec[1];
            }
        }
        else
        {
            unsigned int bucketIndex = 
                (item.featureAttrArray[i] - featureVec[i].min) / 
                featureVec[i].bucketSize;
            
            if (bucketIndex >= featureVec[i].numBuckets)
                bucketIndex = featureVec[i].numBuckets - 1;

            if (node->childrenVec[bucketIndex] == nullptr)
                break;
            else
                node = node->childrenVec[bucketIndex];
        }
    }

    return node->classIndex;
}
