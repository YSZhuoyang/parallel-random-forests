
#include "Classifier.h"


Classifier::Classifier()
{

}

Classifier::~Classifier()
{

}


void Classifier::Train(
    const vector<Item>& iv, 
    const vector<NumericAttr>& fv, 
    const vector<char*>& cv )
{
    classVec = cv;
    featureVec = fv;

    // Randomly select features and build trees




    treeBuilder.BuildTree( iv, fv, cv );
    root = treeBuilder.GetRoot();
}

void Classifier::Classify( const vector<Item>& iv )
{
    if (classVec.empty())
    {
        printf( "Please train the model first" );
        return;
    }

    unsigned int correctCounter = 0;
    unsigned int totalNumber = iv.size();

    for (const Item& item : iv)
    {
        if (Classify( item ) == item.classIndex) correctCounter++;
    }

    float correctRate = (float) correctCounter / (float) totalNumber;
    float incorrectRate = 1 - correctRate;

    printf( "Correct rate: %f\n", correctRate );
    printf( "Incorrect rate: %f\n", incorrectRate );
}

int Classifier::Classify( const Item& item )
{
    TreeNode* node = root;

    while (node != nullptr && !node->childrenVec.empty())
    {
        unsigned int i = node->featureIndex;

        // If there are only 2 buckets, then classify them into:
        // a group of 0, and a group of greater than 0.
        if (featureVec[i].numBuckets == 2)
        {
            if (item.featureAttrArray[i] <= 0)
            {
                node = node->childrenVec[0];
            }
            else
            {
                node = node->childrenVec[1];
            }
        }
        else
        {
            unsigned int sizeOfRange = 
                featureVec[i].max - featureVec[i].min + 1;
            float bucketSize = 
                (float) sizeOfRange / (float) featureVec[i].numBuckets;

            unsigned int bucketIndex = 
                (item.featureAttrArray[i] - featureVec[i].min) / bucketSize;
            node = node->childrenVec[bucketSize];
        }
    }

    if (node == nullptr)
        return -1;
    else
        return node->classIndex;
}
