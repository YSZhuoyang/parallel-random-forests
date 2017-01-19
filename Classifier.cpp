
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
    if (rootArr != nullptr)
    {
        for (unsigned int i = 0; i < numTrees; i++)
            treeBuilder.DestroyNode( rootArr[i] );
        free( rootArr );
        rootArr = nullptr;
    }

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
    if (NUM_TREES % numMpiNodes > 0)
    {
        if (mpiNodeId == numMpiNodes - 1)
            numTrees = NUM_TREES - numTrees * (numMpiNodes - 1);
        else
	    numTrees = NUM_TREES / numMpiNodes + 1;
    }
    else
        numTrees = NUM_TREES / numMpiNodes;

    rootArr = (TreeNode*) malloc( numTrees * sizeof( TreeNode ) );
    treeBuilder.Init( fv, cv, instanceTable, numInstances );

    printf( "Node %d constructed %u trees.\n", mpiNodeId, numTrees );

    // Seed randomizer based on mpi node id
    srand( mpiNodeId + 1 );

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
        for (unsigned int treeId = 0; treeId < numTrees; treeId++)
        {
            rootArr[treeId] = treeBuilder.BuildTree( NUM_FEATURES_PER_TREE );
            //treeBuilder.PrintTree( rootArr[treeId], 0 );
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

inline void Classifier::Classify(
    const Instance& instance, 
    unsigned int* votes, 
    const unsigned int instId )
{
    unsigned short numClasses = classVec.size();

    for (unsigned int treeId = 0; treeId < numTrees; treeId++)
    {
        TreeNode node = rootArr[treeId];
        if (node.empty) continue;

        while (node.childrenArr != nullptr)
        {
            // 2 children by default:
            // one group having feature value smaller than threshold, 
            // another group having feature value greater than threshold.
            unsigned int childId = (unsigned int)
                (instance.featureAttrArray[node.featureIndex] >= node.threshold);
            if (node.childrenArr[childId].empty) break;
            else node = node.childrenArr[childId];
        }

        votes[instId * numClasses + node.classIndex]++;
    }
}
