
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

    unsigned int numFeatures = fv.size();
    printf( "Num features: %d\n", numFeatures );
    
    /******************** Init tree constructer ********************/

    // Build a number of trees each having the same number of features.
    rootVec.reserve( NUM_TREES );
    treeBuilder.Init( fv, cv, iv );

    time_t start,end;
    double dif;
    time( &start );

    #pragma omp parallel
    {
        if (omp_get_thread_num() == 0)
            printf( "There're %d threads running.\n", omp_get_num_threads() );

        #pragma omp for schedule(dynamic)
        for (unsigned int treeIndex = 0; treeIndex < NUM_TREES; treeIndex++)
        {
            treeBuilder.BuildTree( NUM_FEATURES_PER_TREE );
            #pragma omp critical
            rootVec.push_back( treeBuilder.GetRoot() );
            //treeBuilder.PrintTree( treeBuilder.GetRoot(), 0 );
        }
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
    Item instance = Tokenize( str, featureVec );
    int classIndex = Classify( instance );

    free( instance.featureAttrArray );
    instance.featureAttrArray = nullptr;

    printf( "Labeled with: %s\n", cv[classIndex] );

    return cv[classIndex];
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

    #pragma omp parallel for reduction(+: correctCounter) schedule(dynamic)
    for (unsigned int i = 0; i < totalNumber; i++)
        if (Classify( iv[i] ) == iv[i].classIndex) correctCounter++;

    float correctRate = (float) correctCounter / (float) totalNumber;
    float incorrectRate = 1.0f - correctRate;

    printf( "Correct rate: %f\n", correctRate );
    printf( "Incorrect rate: %f\n", incorrectRate );
}

int Classifier::Classify( const Item& instance )
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
            if (instance.featureAttrArray[i] <= node->threshold)
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

        votes[node->classIndex]++;
    }

    unsigned short classIndex = getIndexOfMax( votes, numClasses );

    free( votes );
    votes = nullptr;

    return classIndex;
}
