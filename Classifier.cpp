
#include "Classifier.h"


Classifier::Classifier(
    const vector<NumericAttr>& fv, 
    const vector<char*>& cv )
{
    classVec = cv;
    featureVec = fv;

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


void Classifier::Train( const vector<Item>& iv )
{
    /******************* Prepare buffer *******************/
    unsigned int numFeatures;
    unsigned int numClasses;
    unsigned int numItems;

    unsigned int* randomIndices;
    unsigned int featureArrSize;
    int* attrValueBuff;
    unsigned short* classIndexBuff;

    /************ Broadcast data to other nodes ***********/
    if (mpiNodeId == MPI_ROOT_ID)
    {
        numFeatures = featureVec.size();
        numClasses = classVec.size();
        numItems = iv.size();

        // Randomly select features and build trees.
        // Generate an ordered index container, and disorder it.
        randomIndices = 
            (unsigned int*) malloc( numFeatures * sizeof( unsigned int ) );
        for (unsigned int i = 0; i < numFeatures; i++) randomIndices[i] = i;
        randomizeArray( randomIndices, numFeatures );

        // Convert data to the form can be transfered by MPI
        featureArrSize = numFeatures * sizeof( int );
        attrValueBuff = (int*) malloc( numItems * featureArrSize );
        classIndexBuff = 
            (unsigned short*) malloc( numItems * sizeof( unsigned short ) );

        // To be parallelized by openmp
        for (unsigned int i = 0; i < numItems; i++)
        {
            unsigned int offset = numFeatures;
            memcpy( attrValueBuff + offset, iv[i].featureAttrArray, featureArrSize );
            classIndexBuff[i] = iv[i].classIndex;
        }
    }

    // Need to check error code.
    MPI_Bcast( &featureArrSize, 1, MPI_UNSIGNED, 
        MPI_ROOT_ID, MPI_COMM_WORLD );
    MPI_Bcast( &numClasses, 1, MPI_UNSIGNED, 
        MPI_ROOT_ID, MPI_COMM_WORLD );
    MPI_Bcast( &numFeatures, 1, MPI_UNSIGNED, 
        MPI_ROOT_ID, MPI_COMM_WORLD );
    MPI_Bcast( &numItems, 1, MPI_UNSIGNED, 
        MPI_ROOT_ID, MPI_COMM_WORLD );
    MPI_Barrier( MPI_COMM_WORLD );

    if (mpiNodeId != MPI_ROOT_ID)
    {
        randomIndices = 
            (unsigned int*) malloc( numFeatures * sizeof( unsigned int ) );
        attrValueBuff = (int*) malloc( numItems * featureArrSize );
        classIndexBuff = 
            (unsigned short*) malloc( numItems * sizeof( unsigned short ) );
    }

    MPI_Bcast( randomIndices, numFeatures, 
        MPI_UNSIGNED, MPI_ROOT_ID, MPI_COMM_WORLD );
    MPI_Bcast( attrValueBuff, numItems * numFeatures, 
        MPI_INT, MPI_ROOT_ID, MPI_COMM_WORLD );
    MPI_Bcast( classIndexBuff, numItems, 
        MPI_INT, MPI_ROOT_ID, MPI_COMM_WORLD );
    MPI_Barrier( MPI_COMM_WORLD );

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

    // Build a number of trees each having a fixed number of features.
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

        // Free data copy
        for (Item& item : itemVecCopy) free( item.featureAttrArray );
        itemVecCopy.clear();
    }

    free( randomIndices );
    randomIndices = nullptr;
}

void Classifier::Classify( const vector<Item>& iv )
{
    if (mpiNodeId == MPI_ROOT_ID && classVec.empty())
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
    unsigned int numFeatures;
    unsigned int numClasses;
    unsigned int numItems;
    unsigned int featureArrSize;
    int* attrValueBuff;
    unsigned short* classIndexBuff;

    /************ Broadcast data to other nodes ***********/
    if (mpiNodeId == MPI_ROOT_ID)
    {
        numFeatures = featureVec.size();
        numClasses = classVec.size();
        numItems = iv.size();
        featureArrSize = numFeatures * sizeof( int );

        attrValueBuff = (int*) malloc( numItems * featureArrSize );
        classIndexBuff = 
            (unsigned short*) malloc( numItems * sizeof( unsigned short ) );

        // To be parallelized with openmp
        for (unsigned int i = 0; i < numItems; i++)
        {
            unsigned int offset = numFeatures;
            memcpy( attrValueBuff + offset, iv[i].featureAttrArray, featureArrSize );
            classIndexBuff[i] = iv[i].classIndex;
        }
    }

    // Need to check error code.
    MPI_Bcast( &featureArrSize, 1, MPI_UNSIGNED, 
        MPI_ROOT_ID, MPI_COMM_WORLD );
    MPI_Bcast( &numClasses, 1, MPI_UNSIGNED, 
        MPI_ROOT_ID, MPI_COMM_WORLD );
    MPI_Bcast( &numFeatures, 1, MPI_UNSIGNED, 
        MPI_ROOT_ID, MPI_COMM_WORLD );
    MPI_Bcast( &numItems, 1, MPI_UNSIGNED, 
        MPI_ROOT_ID, MPI_COMM_WORLD );
    MPI_Barrier( MPI_COMM_WORLD );

    if (mpiNodeId != MPI_ROOT_ID)
    {
        attrValueBuff = (int*) malloc( numItems * featureArrSize );
        classIndexBuff = 
            (unsigned short*) malloc( numItems * sizeof( unsigned short ) );
    }

    MPI_Bcast( attrValueBuff, numItems * numFeatures, 
        MPI_INT, MPI_ROOT_ID, MPI_COMM_WORLD );
    MPI_Bcast( classIndexBuff, numItems, 
        MPI_INT, MPI_ROOT_ID, MPI_COMM_WORLD );
    MPI_Barrier( MPI_COMM_WORLD );

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

    /***************** Do reduction here ******************/



    
    unsigned int correctCounter = 0;
    
    if (mpiNodeId == MPI_ROOT_ID)
    {
        for (const Item& item : iv)
            if (Classify( item, numClasses ) == item.classIndex)
                correctCounter++;
    }
    else
    {
        for (const Item& item : itemVecCopy)
            if (Classify( item, numClasses ) == item.classIndex)
                correctCounter++;

        // Free data copy
        for (Item& item : itemVecCopy) free( item.featureAttrArray );
        itemVecCopy.clear();
    }

    float correctRate = (float) correctCounter / (float) numItems;
    float incorrectRate = 1.0f - correctRate;

    printf( "Correct rate: %f\n", correctRate );
    printf( "Incorrect rate: %f\n", incorrectRate );
}

int Classifier::Classify(
    const Item& item, 
    const unsigned int numClasses )
{
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
