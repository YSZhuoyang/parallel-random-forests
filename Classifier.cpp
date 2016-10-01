
#include "Classifier.h"


Classifier::Classifier()
{
    
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
    unsigned int numFeatures = fv.size();
    unsigned int numClasses = cv.size();
    unsigned int numItems = iv.size();

    printf( "Num features: %d\n", numFeatures );

    // Randomly select features and build trees.
    // Generate an ordered index container, and disorder it.
    unsigned int* randomIndices = 
        (unsigned int*) malloc( numFeatures * sizeof( unsigned int ) );
    for (unsigned int i = 0; i < numFeatures; i++) randomIndices[i] = i;
    randomizeArray( randomIndices, numFeatures, numFeatures );

    /******* Prepare buffer and init MPI processes *******/
    unsigned int featureArrSize = numFeatures * sizeof( int );
    int* attrValueBuff = (int*) malloc( numItems * featureArrSize );
    unsigned short* classIndexBuff = 
        (unsigned short*) malloc( numItems * sizeof( unsigned short ) );

    // Init MPI
    MPI_Initialized( &mpiInitialized );
    if (!mpiInitialized) MPI_Init( nullptr, nullptr );

    int numMpiNodes;
    int mpiNodeId;

    MPI_Comm_size( MPI_COMM_WORLD, &numMpiNodes );
    MPI_Comm_rank( MPI_COMM_WORLD, &mpiNodeId );

    printf( "There are %d nodes doing computation.\n", numMpiNodes );
    printf( "Id of this node is: %d.\n", mpiNodeId );

    /************ Broadcast data to other nodes ***********/
    // Move it to the front of init mpi?
    if (mpiNodeId == MPI_ROOT_ID)
    {
        // To be parallelized with openmp
        for (unsigned int i = 0; i < numItems; i++)
        {
            unsigned int offset = numFeatures;
            memcpy( attrValueBuff + offset, iv[i].featureAttrArray, featureArrSize );
            classIndexBuff[i] = iv[i].classIndex;
        }
    }

    MPI_Bcast( &numClasses, 1, MPI_UNSIGNED, 
        MPI_ROOT_ID, MPI_COMM_WORLD );
    MPI_Bcast( &numFeatures, 1, MPI_UNSIGNED, 
        MPI_ROOT_ID, MPI_COMM_WORLD );
    MPI_Bcast( &numItems, 1, MPI_UNSIGNED, 
        MPI_ROOT_ID, MPI_COMM_WORLD );
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Bcast( randomIndices, numFeatures, 
        MPI_UNSIGNED, MPI_ROOT_ID, MPI_COMM_WORLD );
    MPI_Bcast( attrValueBuff, numItems * numFeatures, 
        MPI_INT, MPI_ROOT_ID, MPI_COMM_WORLD );
    MPI_Bcast( classIndexBuff, numItems, 
        MPI_INT, MPI_ROOT_ID, MPI_COMM_WORLD );
    MPI_Barrier(MPI_COMM_WORLD);

    /**************** Convert back data format ****************/
    vector<Item> itemVecCopy;

    if (mpiNodeId != MPI_ROOT_ID)
    {
        itemVecCopy.reserve( numItems );

        for (unsigned int i = 0; i < numItems; i++)
        {
            Item item;
            item.classIndex = classIndexBuff[i];
            item.featureAttrArray = 
                (int*) malloc( numFeatures * sizeof( int ) );
            memcpy( item.featureAttrArray, 
                attrValueBuff + i * numFeatures, 
                numFeatures * sizeof( int ) );
            
            itemVecCopy.push_back( item );
        }
    }

    free( attrValueBuff );
    free( classIndexBuff );

    /*MPI_Datatype MPI_Item;
    MPI_Datatype fieldTypes[2] = { MPI_UNSIGNED_SHORT, MPI_INT };
    MPI_Aint offsets[2];
    MPI_Aint baseAddr;
    int blockLength[2] = { 1, (int) numFeatures };

    // Compute offsets
    MPI_Get_address( &iv[0], &offsets[0] );
    MPI_Get_address( &iv[0].featureAttrArray, &offsets[1] );

    // Set offsets start from 0
    baseAddr = offsets[0];
    for (int i = 0; i < 2; i++) offsets[i] -= baseAddr;

    //MPI_Type_create_struct( 2, blockLength, offsets, fieldTypes, &MPI_Item );
    MPI_Type_create_hindexed( 2, blockLength, offsets, fieldTypes, &MPI_Item );
    MPI_Type_commit( &MPI_Item );*/

    /*****************************************************/

    // Build a number of trees each of which having 10 features.
    // What if numFeatures is 51 ?
    unsigned int numTrees = numFeatures / NUM_FEATURES_PER_TREE;
    unsigned int chunkSize = numTrees / numMpiNodes;
    unsigned int startIndex = mpiNodeId * chunkSize;

    printf( "Node %d constructed %u trees.\n", mpiNodeId, chunkSize );

    rootVec.reserve( chunkSize );
    treeBuilder.Init( numClasses, NUM_FEATURES_PER_TREE );

    if (mpiNodeId == MPI_ROOT_ID)
    {
        for (unsigned int i = 0; i < chunkSize; i++)
        {
            unsigned int treeIndex = startIndex + i;
            unsigned int* featureIndexArr = (unsigned int*) 
                malloc( NUM_FEATURES_PER_TREE * sizeof( unsigned int ) );
            memcpy( featureIndexArr, 
                randomIndices + treeIndex * NUM_FEATURES_PER_TREE, 
                NUM_FEATURES_PER_TREE * sizeof( unsigned int ) );

            treeBuilder.BuildTree( iv, featureIndexArr );
            rootVec.push_back( treeBuilder.GetRoot() );
        }
    }
    else
    {
        for (unsigned int i = 0; i < chunkSize; i++)
        {
            unsigned int treeIndex = startIndex + i;
            unsigned int* featureIndexArr = (unsigned int*) 
                malloc( NUM_FEATURES_PER_TREE * sizeof( unsigned int ) );
            memcpy( featureIndexArr, 
                randomIndices + treeIndex * NUM_FEATURES_PER_TREE, 
                NUM_FEATURES_PER_TREE * sizeof( unsigned int ) );

            treeBuilder.BuildTree( itemVecCopy, featureIndexArr );
            rootVec.push_back( treeBuilder.GetRoot() );
        }
    }

    free( randomIndices );
    randomIndices = nullptr;

    /*******************************/
    if (mpiNodeId != 0)
    {
        for (Item& item : itemVecCopy)
            free( item.featureAttrArray );
        
        itemVecCopy.clear();
    }

    //MPI_Type_free( &MPI_Item );
    /*******************************/
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
            if (item.featureAttrArray[i] <= node->threshold)
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
