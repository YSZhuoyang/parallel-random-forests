
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
    const Instance* instanceTable,
    const vector<NumericAttr>& fv,
    const vector<char*>& cv,
    const unsigned int numInstances )
{
    classVec = cv;
    featureVec = fv;
    
    /******************** Init tree constructer ********************/
    // Chunk size need to be able to be divided by number of mpi processes.
    unsigned int chunkSize = NUM_TREES / numMpiNodes;
    rootVec.reserve( chunkSize );
    treeBuilder.Init( fv, cv, instanceTable, numInstances );

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

void Classifier::Classify(
    const Instance* instanceTable,
    const unsigned int numInstances )
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
    unsigned int correctCounter = 0;
    unsigned int* votes = (unsigned int*) 
        calloc( numClasses * numInstances, sizeof( unsigned int ) );

    #pragma omp parallel for schedule(dynamic)
    for (unsigned int i = 0; i < numInstances; i++)
        Classify( instanceTable[i], votes, i );
    
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
            if (predictedClassIndex == instanceTable[i].classIndex)
                correctCounter++;
        }

        double correctRate = (double) correctCounter / (double) numInstances;
        double incorrectRate = 1.0 - correctRate;

        printf( "Correct rate: %f\n", correctRate );
        printf( "Incorrect rate: %f\n", incorrectRate );
    }

    free( votes );
    votes = nullptr;
}

void Classifier::Classify(
    const Instance& instance, 
    unsigned int* votes, 
    const unsigned int index )
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
            unsigned int childId =
                (unsigned int) (instance.featureAttrArray[i] >= node->threshold);
            if (node->childrenVec[childId] == nullptr) break;
            else node = node->childrenVec[childId];
        }

        votes[index * numClasses + node->classIndex]++;
    }
}
