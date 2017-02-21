
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

    MiniInstance* miniInstanceArr =
        (MiniInstance*) malloc( numInstancesTotal * sizeof( MiniInstance ) );
    unsigned int* initialClassDist = 
        (unsigned int*) calloc( numClasses, sizeof( unsigned int ) );
    for (unsigned int i = 0; i < numInstancesTotal; i++)
    {
        // Get overall distribution
        initialClassDist[instanceTable[i].classIndex]++;
        // Init data indices and copy class indices
        miniInstanceArr[i].featureIndex = i;
        miniInstanceArr[i].classIndex = instanceTable[i].classIndex;
    }
    
    TreeNode* root = Split(
        miniInstanceArr,
        featureIndexArray,
        initialClassDist,
        numInstancesTotal,
        0 );

    free( initialClassDist );
    initialClassDist = nullptr;
    free( miniInstanceArr );
    miniInstanceArr = nullptr;
    free( featureIndexArray );
    featureIndexArray = nullptr;

    return root;
}

TreeNode* TreeBuilder::Split(
    MiniInstance* miniInstanceArr,
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
    classDistArr[0] = ( unsigned int* )
        malloc( NUM_CHILD_NUMERICAL * numClasses * sizeof( unsigned int ) );
    classDistArr[1] = classDistArr[0] + numClasses;

    unsigned int* selectedClassDistArr[NUM_CHILD_NUMERICAL];
    selectedClassDistArr[0] = ( unsigned int* )
        malloc( NUM_CHILD_NUMERICAL * numClasses * sizeof( unsigned int ) );
    selectedClassDistArr[1] = selectedClassDistArr[0] + numClasses;

    // Store sorted values of that feature with indices
    MiniInstance* selectedMiniInstanceArr =
        (MiniInstance*) malloc( numInstances * sizeof( MiniInstance ) );

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
            miniInstanceArr[i].featureValue =
                instanceTable[miniInstanceArr[i].featureIndex].
                    featureAttrArray[randFeaIndex];
        sort(
            miniInstanceArr,
            miniInstanceArr + numInstances,
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
            
            classDistArr[0][miniInstanceArr[preCandidateId].classIndex]++;
            classDistArr[1][miniInstanceArr[preCandidateId].classIndex]--;

            if (miniInstanceArr[preCandidateId].featureValue <
                miniInstanceArr[candidateId].featureValue)
            {
                double splitThreshold =
                    (miniInstanceArr[preCandidateId].featureValue + 
                    miniInstanceArr[candidateId].featureValue) / 2.0;
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
                            selectedMiniInstanceArr,
                            miniInstanceArr,
                            numInstances * sizeof( MiniInstance ) );

                        selectedFeaIndex = randFeaIndex;
                        featureIndexStored = true;
                    }

                    memmove(
                        selectedClassDistArr[0],
                        classDistArr[0],
                        NUM_CHILD_NUMERICAL * numClasses * sizeof( unsigned int ) );

                    selectedChildSizeArr[0] = childSizeArr[0];
                    selectedChildSizeArr[1] = childSizeArr[1];
                    // giniImpurityMax = giniImpurity;
                    infoGainMax = infoGain;
                    selectedThreshold = splitThreshold;

                    if (!gainFound) gainFound = true;
                }
            }
        }
    }

    free( classDistArr[0] );
    
    //printf( "\n----------------------------------------\n");
    //printf( "Height: %d\n", height );

    TreeNode* node;

    // Split threshold not found, 
    // or gini impurity / info gain exceeds threshold,
    // thus have reached leaf node.
    if (!gainFound)
    {
        free( selectedClassDistArr[0] );
        selectedClassDistArr[0] = nullptr;
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
            // Consider NUM_CHILD_NUMERICAL is 2, childId is either 0 or 1.
            MiniInstance* childMiniInstanceArr =
                selectedMiniInstanceArr + childId * selectedChildSizeArr[0];
            
            node->childrenArr[childId] = Split(
                childMiniInstanceArr,
                featureIndexArray,
                selectedClassDistArr[childId],
                selectedChildSizeArr[childId],
                height );

            if (node->childrenArr[childId] == nullptr)
                emptyChildFound = true;
        }

        free( selectedClassDistArr[0] );
        selectedClassDistArr[0] = nullptr;

        if (emptyChildFound) LabelNode( node, parentClassDist );
    }

    free( selectedMiniInstanceArr );
    selectedMiniInstanceArr = nullptr;

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
