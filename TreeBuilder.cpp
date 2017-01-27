
#include "TreeBuilder.h"


TreeBuilder::TreeBuilder()
{
    
}

TreeBuilder::~TreeBuilder()
{
    
}

void TreeBuilder::Init(
    const vector<NumericAttr>& fv,
    const vector<char*>& cv,
    const Instance* it,
    const unsigned int numInstances )
{
    featureVec = fv;
    classVec = cv;
    instanceTable = it;
    numInstancesTotal = numInstances;
    numFeaturesTotal = featureVec.size();
    numClasses = classVec.size();
}

TreeNode* TreeBuilder::BuildTree( const unsigned int numFeaToSelect )
{
    numFeaturesToSelect = numFeaToSelect;
    
    unsigned int* featureIndexArray = 
        (unsigned int*) malloc( numFeaturesTotal * sizeof( unsigned int ) );
    for (unsigned int i = 0; i < numFeaturesTotal; i++)
        featureIndexArray[i] = i;

    ValueIndexTuple* valueIndexTupleArr =
        (ValueIndexTuple*) malloc( numInstancesTotal * sizeof( ValueIndexTuple ) );
    unsigned int* initialClassDist = 
        (unsigned int*) calloc( numClasses, sizeof( unsigned int ) );
    for (unsigned int i = 0; i < numInstancesTotal; i++)
    {
        // Get overall distribution
        initialClassDist[instanceTable[i].classIndex]++;
        // Init data indices and copy class indices
        valueIndexTupleArr[i].featureIndex = i;
        valueIndexTupleArr[i].classIndex = instanceTable[i].classIndex;
    }
    
    TreeNode* root = Split(
        valueIndexTupleArr,
        featureIndexArray,
        initialClassDist,
        numInstancesTotal,
        0 );

    free( initialClassDist );
    initialClassDist = nullptr;
    free( valueIndexTupleArr );
    valueIndexTupleArr = nullptr;
    free( featureIndexArray );
    featureIndexArray = nullptr;

    return root;
}

TreeNode* TreeBuilder::Split(
    ValueIndexTuple* valueIndexTupleArr,
    unsigned int* featureIndexArray,
    const unsigned int* parentClassDist,
    const unsigned int numInstances,
    unsigned int height )
{
    // double giniImpurityMax = 0;
    double infoGainMax = 0;

    // The node is too small thus it is ignored.
    if (numInstances < MIN_NODE_SIZE) return nullptr;

    // The node is small, make it a leaf node
    // or node is pure.
    if (numInstances < MIN_NODE_SIZE_TO_SPLIT)
    {
        TreeNode* leaf = new TreeNode;
        leaf->childrenArr = nullptr;
        LabelNode( leaf, parentClassDist );
        
        return leaf;
    }

    // Compute entropy of parent node.
    double entropyParent = ComputeEntropy( parentClassDist, numInstances );
    // double giniParent = ComputeGini( parentClassDist, numInstances );

    // Parent node is pure.
    // if (giniParent <= 0.0)
    if (entropyParent <= 0.0)
    {
        TreeNode* leaf = new TreeNode;
        leaf->childrenArr = nullptr;
        LabelNode( leaf, parentClassDist );

        return leaf;
    }

    unsigned int selectedFeaIndex;
    double selectedThreshold;

    unsigned int childSizeArr[NUM_CHILD_NUMERICAL];
    unsigned int selectedChildSizeArr[NUM_CHILD_NUMERICAL];

    // Init child class distribution vector
    unsigned int* classDistArr[NUM_CHILD_NUMERICAL];
    unsigned int* selectedClassDistArr[NUM_CHILD_NUMERICAL];
    for (unsigned int childId = 0; childId < NUM_CHILD_NUMERICAL; childId++)
    {
        classDistArr[childId] = ( unsigned int* )
            malloc( numClasses * sizeof( unsigned int ) );
        selectedClassDistArr[childId] = ( unsigned int* )
            malloc( numClasses * sizeof( unsigned int ) );
    }

    // Store sorted values of that feature with indices
    ValueIndexTuple* selectedValueIndexTupleArr =
        (ValueIndexTuple*) malloc( numInstances * sizeof( ValueIndexTuple ) );

    unsigned int numRestFeaToSelect = numFeaturesToSelect;
    unsigned int numRestFea = numFeaturesTotal;
    bool gainFound = false;

    // Find the best split feature and threshold
    while ((numRestFeaToSelect > 0 || !gainFound) && numRestFea > 0)
    {
        // Sample (note max of rand() is around 32000)
        unsigned int randPos = rand() % numRestFea;
        unsigned int randFeaIndex = featureIndexArray[randPos];
        // Swap
        featureIndexArray[randPos] = featureIndexArray[--numRestFea];
        featureIndexArray[numRestFea] = randFeaIndex;

        if (numRestFeaToSelect > 0) numRestFeaToSelect--;

        // Get all values of that feature with indices and sort them.
        for (unsigned int i = 0; i < numInstances; i++)
            valueIndexTupleArr[i].featureValue =
                instanceTable[valueIndexTupleArr[i].featureIndex].
                    featureAttrArray[randFeaIndex];
        sort(
            valueIndexTupleArr,
            valueIndexTupleArr + numInstances,
            Compare );

        // Reset split index and child class distribution
        memset( classDistArr[0], 0, numClasses * sizeof( unsigned int ) );
        memmove(
            classDistArr[1],
            parentClassDist,
            numClasses * sizeof( unsigned int ) );

        bool featureIndexStored = false;

        // Find the best split threshold
        for (unsigned int candidateId = 1;
            candidateId < numInstances; candidateId++)
        {
            unsigned int preCandidateId = candidateId - 1;
            
            classDistArr[0][valueIndexTupleArr[preCandidateId].classIndex]++;
            classDistArr[1][valueIndexTupleArr[preCandidateId].classIndex]--;

            if (valueIndexTupleArr[preCandidateId].featureValue <
                valueIndexTupleArr[candidateId].featureValue)
            {
                double splitThreshold =
                    (valueIndexTupleArr[preCandidateId].featureValue + 
                    valueIndexTupleArr[candidateId].featureValue) / 2.0;
                childSizeArr[0] = candidateId;
                childSizeArr[1] = numInstances - candidateId;

                // double giniImpurity = giniParent;
                double infoGain = entropyParent;
                
                // Compute entropy of children
                for (unsigned int childId = 0; childId < NUM_CHILD_NUMERICAL; childId++)
                {
                    double numChildren = childSizeArr[childId];
                    double entropyChild =
                        ComputeEntropy( classDistArr[childId], numChildren );
                    infoGain -= numChildren / (double) numInstances * entropyChild;
                    // double giniChild =
                    //     ComputeGini( classDistArr[childId], numChildren );
                    // giniImpurity -= numChildren / (double) numInstances * giniChild;
                }

                // Get max split outcome and related feature
                // if (giniImpurityMax < giniImpurity)
                if (infoGainMax < infoGain)
                {
                    if (!featureIndexStored)
                    {
                        memmove(
                            selectedValueIndexTupleArr,
                            valueIndexTupleArr,
                            numInstances * sizeof( ValueIndexTuple ) );

                        selectedFeaIndex = randFeaIndex;
                        featureIndexStored = true;
                    }

                    for (unsigned int childId = 0; childId < NUM_CHILD_NUMERICAL; childId++)
                    {
                        selectedChildSizeArr[childId] = childSizeArr[childId];
                        memmove(
                            selectedClassDistArr[childId],
                            classDistArr[childId],
                            numClasses * sizeof( unsigned int ) );
                    }

                    // giniImpurityMax = giniImpurity;
                    infoGainMax = infoGain;
                    selectedThreshold = splitThreshold;

                    if (!gainFound) gainFound = true;
                }
            }
        }
    }

    for (unsigned int childId = 0; childId < NUM_CHILD_NUMERICAL; childId++)
        free( classDistArr[childId] );
    //printf( "\n----------------------------------------\n");
    //printf( "Height: %d\n", height );

    TreeNode* node;

    // Split threshold not found, 
    // or gini impurity / info gain exceeds threshold,
    // thus have reached leaf node.
    if (!gainFound)
    {
        for (unsigned int childId = 0; childId < NUM_CHILD_NUMERICAL; childId++)
        {
            free( selectedClassDistArr[childId] );
            selectedClassDistArr[childId] = nullptr;
        }

        node = nullptr;
    }
    // Split node
    else
    {
        //printf( "Feature selected: %s\n", featureVec[selectedFeaIndex].name );
        //printf( "Gini of parent: %f\n", giniParent );
        //printf( "Max Gini split get: %f\n", giniImpurityMax );
        
        node = new TreeNode;
        node->featureIndex = selectedFeaIndex;
        node->threshold    = selectedThreshold;
        node->childrenArr  =
            (TreeNode**) malloc( NUM_CHILD_NUMERICAL * sizeof( TreeNode* ) );

        height++;

        bool emptyChildFound = false;

        // Split children
        for (unsigned int childId = 0; childId < NUM_CHILD_NUMERICAL; childId++)
        {
            ValueIndexTuple* childValueIndexTupleArr = (ValueIndexTuple*)
                malloc( selectedChildSizeArr[childId] * sizeof( ValueIndexTuple ) );
            // Consider NUM_CHILD_NUMERICAL is 2, childId is either 0 or 1.
            ValueIndexTuple* offset = selectedValueIndexTupleArr +
                ((childId) ? selectedChildSizeArr[0] : 0);
            memmove(
                childValueIndexTupleArr,
                offset,
                selectedChildSizeArr[childId] * sizeof( ValueIndexTuple ) );
            
            node->childrenArr[childId] = Split(
                childValueIndexTupleArr,
                featureIndexArray,
                selectedClassDistArr[childId],
                selectedChildSizeArr[childId],
                height );

            if (node->childrenArr[childId] == nullptr)
                emptyChildFound = true;

            free( childValueIndexTupleArr );
            childValueIndexTupleArr = nullptr;
            free( selectedClassDistArr[childId] );
            selectedClassDistArr[childId] = nullptr;
        }

        if (emptyChildFound) LabelNode( node, parentClassDist );
    }

    free( selectedValueIndexTupleArr );
    selectedValueIndexTupleArr = nullptr;

    return node;
}

void TreeBuilder::PrintTree( const TreeNode* node, unsigned int h )
{
    if (node == nullptr) return;

    for (unsigned int i = 0; i <= h; i++) printf( "-" );
    printf( "feature: %s, ", featureVec[node->featureIndex].name );
    printf( "threshold: %f\n", node->threshold );

    for (unsigned int childId = 0; childId < NUM_CHILD_NUMERICAL; childId++)
        PrintTree( node->childrenArr[childId], h + 1 );
}

inline double TreeBuilder::ComputeEntropy(
    const unsigned int* classDistribution,
    const unsigned int numInstances )
{
    if (numInstances == 0) return 0.0;

    double entropy = 0.0;

    for (unsigned short i = 0; i < numClasses; i++)
        if (classDistribution[i] > 0 && classDistribution[i] < numInstances)
        {
            double temp = (double) classDistribution[i] / numInstances;
            entropy -= temp * log2( temp );
        }

    return entropy;
}

inline double TreeBuilder::ComputeGini(
    const unsigned int* classDistribution,
    const unsigned int numInstances )
{
    if (numInstances == 0) return 0.0;

    double gini = 1.0;

    for (unsigned short i = 0; i < numClasses; i++)
    {
        double temp = (double) classDistribution[i] / numInstances;
        gini -= temp * temp;
    }

    return gini;
}

inline void TreeBuilder::LabelNode(
    TreeNode* node,
    const unsigned int* classDistribution )
{
    if (node == nullptr || classDistribution == nullptr)
        return;

    // Select the class of the largest class group.
    node->classIndex = getIndexOfMax( classDistribution, numClasses );
}

void TreeBuilder::DestroyNode( TreeNode* node )
{
    if (node == nullptr) return;

    if (node->childrenArr != nullptr)
        for (unsigned int childId = 0;
            childId < NUM_CHILD_NUMERICAL; childId++)
            DestroyNode( node->childrenArr[childId] );

    free( node->childrenArr );
    node->childrenArr = nullptr;
    delete node;
    node = nullptr;
}
