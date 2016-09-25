
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

unsigned short Classifier::Classify( const Item& item )
{
    TreeNode* node = root;

    while (node != nullptr && !node->childrenVec.empty())
    {
        
    }
}
