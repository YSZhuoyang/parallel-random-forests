
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
    const Instance* instanceTable,
    const vector<NumericAttr>& fv,
    const vector<char*>& cv,
    const unsigned int numInstances )
{
    classVec = cv;
    featureVec = fv;

    unsigned int numFeatures = fv.size();
    printf( "Num features: %d\n", numFeatures );
    
    /******************** Init tree constructer ********************/
    rootVec.reserve( NUM_TREES );
    treeBuilder.Init( fv, cv, instanceTable, numInstances );

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
    Instance instance = Tokenize( str, featureVec );
    unsigned short classIndex = Classify( instance );

    free( instance.featureAttrArray );
    instance.featureAttrArray = nullptr;

    printf( "Labeled with: %s\n", cv[classIndex] );

    return cv[classIndex];
}

void Classifier::Classify(
    const Instance* instanceTable,
    const unsigned int numInstances )
{
    if (classVec.empty())
    {
        printf( "Please train the model first.\n" );
        return;
    }

    unsigned int correctCounter = 0;

    #pragma omp parallel for reduction (+: correctCounter) schedule(dynamic)
    for (unsigned int instId = 0; instId < numInstances; instId++)
        if (Classify( instanceTable[instId] ) ==
            instanceTable[instId].classIndex)
            correctCounter++;

    double correctRate = (double) correctCounter / (double) numInstances;
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
