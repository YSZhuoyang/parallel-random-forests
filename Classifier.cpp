
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
    const vector<Instance>& iv, 
    const vector<NumericAttr>& fv, 
    const vector<char*>& cv )
{
    classVec = cv;
    featureVec = fv;

    unsigned int numFeatures = fv.size();
    printf( "Num features: %d\n", numFeatures );
    
    /******************** Init tree constructer ********************/

    // Build a number of trees each having the same number of features.
    rootVec.reserve( NUM_TREES );
    treeBuilder.Init( fv, cv, iv );

    time_t start,end;
    double dif;
    time( &start );

    for (unsigned int treeIndex = 0; treeIndex < NUM_TREES; treeIndex++)
    {
        treeBuilder.BuildTree( NUM_FEATURES_PER_TREE );
        rootVec.push_back( treeBuilder.GetRoot() );
        // treeBuilder.PrintTree( treeBuilder.GetRoot(), 0 );
    }

    time( &end );
    dif = difftime( end, start );

    printf( "Build forests: time taken is %.2lf seconds.\n", dif );
}

char* Classifier::Analyze(
    const char* str,
    const vector<NumericAttr>& featureVec,
    const vector<char*>& cv )
{
    Instance instance = Tokenize( str, featureVec );
    unsigned short classIndex = Classify( instance );

    free( instance.featureAttrArray );
    instance.featureAttrArray = nullptr;

    printf( "Labeled with: %s\n", cv[classIndex] );

    return cv[classIndex];
}

void Classifier::Classify( const vector<Instance>& iv )
{
    if (classVec.empty())
    {
        printf( "Please train the model first.\n" );
        return;
    }

    unsigned int correctCounter = 0;
    unsigned int totalNumber = iv.size();

    for (const Instance& instance : iv)
        if (Classify( instance ) == instance.classIndex) correctCounter++;

    double correctRate = (double) correctCounter / (double) totalNumber;
    double incorrectRate = 1.0 - correctRate;

    printf( "Correct rate: %f\n", correctRate );
    printf( "Incorrect rate: %f\n", incorrectRate );
}

unsigned short Classifier::Classify( const Instance& instance )
{
    unsigned short numClasses = classVec.size();
    unsigned int* votes = (unsigned int*) 
        calloc( numClasses, sizeof( unsigned int ) );

    for (const TreeNode* node : rootVec)
    {
        if (node == nullptr) continue;

        while (!node->childrenVec.empty())
        {
            unsigned int i = node->featureIndex;

            // 2 buckets by default:
            // one group having feature value smaller than threshold, 
            // another group having feature value greater than threshold.
            unsigned int childId =
                (unsigned int) (instance.featureAttrArray[i] >= node->threshold);
            if (node->childrenVec[childId] == nullptr) break;
            else node = node->childrenVec[childId];
        }

        votes[node->classIndex]++;
    }

    unsigned short classIndex = getIndexOfMax( votes, numClasses );

    free( votes );
    votes = nullptr;

    return classIndex;
}
