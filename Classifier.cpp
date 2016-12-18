
#include "Classifier.h"


Classifier::Classifier()
{
    // Init MPI
    MPI_Initialized( &mpiInitialized );
    if (!mpiInitialized) MPI_Init( nullptr, nullptr );

    MPI_Comm_size( MPI_COMM_WORLD, &numMpiNodes );
    MPI_Comm_rank( MPI_COMM_WORLD, &mpiNodeId );

    if (mpiNodeId == MPI_ROOT_ID)
        printf( "There are %d nodes doing computation.\n", numMpiNodes );
    
    printf( "Id of this node is: %d.\n", mpiNodeId );
}

Classifier::~Classifier()
{
    // Destory trees
    for (TreeNode* root : rootVec) treeBuilder.DestroyNode( root );
    rootVec.clear();

    MPI_Initialized( &mpiInitialized );
    if (mpiInitialized) MPI_Finalize();
}


void Classifier::Train(
    const vector<Item>& iv, 
    const vector<NumericAttr>& fv, 
    const vector<char*>& cv )
{
    classVec = cv;
    featureVec = fv;
    
    /******************** Init tree constructer ********************/
    // Build a number of trees with randomly selected features.
    unsigned int chunkSize = NUM_TREES / numMpiNodes;
    rootVec.reserve( chunkSize );
    treeBuilder.Init( fv, cv, iv );

    printf( "Node %d constructed %u trees.\n", mpiNodeId, chunkSize );

    time_t start,end;
    double dif;
    time( &start );

    #pragma omp parallel
    {
        if (omp_get_thread_num() == 0)
            printf(
                "There're %d threads running on node %d.\n",
                omp_get_num_threads(),
                mpiNodeId );

        #pragma omp for schedule(dynamic)
        for (unsigned int treeIndex = 0; treeIndex < chunkSize; treeIndex++)
        {
            treeBuilder.BuildTree( NUM_FEATURES_PER_TREE );
            #pragma omp critical
            rootVec.push_back( treeBuilder.GetRoot() );
            //treeBuilder.PrintTree( treeBuilder.GetRoot(), 0 );
        }
    }
    
    time( &end );
    dif = difftime( end, start );

    printf( "Node %d builds forests: time taken is %.2lf seconds.\n", mpiNodeId, dif );
}

void Classifier::Classify( const vector<Item>& iv )
{
    if (classVec.empty())
    {
        printf( "Please train the model first.\n" );
        
        MPI_Initialized( &mpiInitialized );
        if (mpiInitialized)
        {
            int err = 0;
            MPI_Abort( MPI_COMM_WORLD, err );
        }

        return;
    }

    /******************* Prepare buffer *******************/
    unsigned short numClasses = classVec.size();
    unsigned int numInstances = iv.size();
    unsigned int correctCounter = 0;
    unsigned int* votes = (unsigned int*) 
        calloc( numClasses * numInstances, sizeof( unsigned int ) );

    #pragma omp parallel for schedule(dynamic)
    for (unsigned int i = 0; i < numInstances; i++)
        Classify( iv[i], votes, i );
    
    /************************** Do reduction *************************/
    if (mpiNodeId == MPI_ROOT_ID)
        CheckMPIErr( MPI_Reduce( MPI_IN_PLACE, votes, numClasses * numInstances, 
            MPI_UNSIGNED, MPI_SUM, 0, MPI_COMM_WORLD ), mpiNodeId );
    else
        CheckMPIErr( MPI_Reduce( votes, nullptr, numClasses * numInstances, 
            MPI_UNSIGNED, MPI_SUM, 0, MPI_COMM_WORLD ), mpiNodeId );
    
    /************************ Compute accuracy ************************/
    if (mpiNodeId == MPI_ROOT_ID)
    {
        #pragma omp parallel for reduction(+: correctCounter) schedule(static)
        for (unsigned int i = 0; i < numInstances; i++)
        {
            unsigned short predictedClassIndex = 
                getIndexOfMax( votes + i * numClasses, numClasses );
            if (predictedClassIndex == iv[i].classIndex)
                correctCounter++;
        }

        float correctRate = (float) correctCounter / (float) numInstances;
        float incorrectRate = 1.0f - correctRate;

        printf( "Correct rate: %f\n", correctRate );
        printf( "Incorrect rate: %f\n", incorrectRate );
    }

    free( votes );
    votes = nullptr;
}

void Classifier::Classify(
    const Item& instance, 
    unsigned int* votes, 
    unsigned int index )
{
    unsigned short numClasses = classVec.size();

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

        votes[index * numClasses + node->classIndex]++;
    }
}
