
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

void TreeBuilder::BuildTree( const unsigned int numFeaToSelect )
{
    numFeaturesToSelect = numFeaToSelect;
    
    unsigned int* featureIndexArray = 
        (unsigned int*) malloc( numFeaturesTotal * sizeof( unsigned int ) );
    for (unsigned int i = 0; i < numFeaturesTotal; i++)
        featureIndexArray[i] = i;

    ValueIndexPair* valueIndexPairArr =
        (ValueIndexPair*) malloc( numInstancesTotal * sizeof( ValueIndexPair ) );
    unsigned int* initialClassDist = 
        (unsigned int*) calloc( numClasses, sizeof( unsigned int ) );
    for (unsigned int i = 0; i < numInstancesTotal; i++)
    {
        // Get overall distribution
        initialClassDist[instanceTable[i].classIndex]++;
        // Init data indices
        valueIndexPairArr[i].featureIndex = i;
    }
    
    root = Split(
        valueIndexPairArr,
        featureIndexArray,
        initialClassDist,
        numInstancesTotal,
        0 );

    free( initialClassDist );
    initialClassDist = nullptr;
    free( valueIndexPairArr );
    valueIndexPairArr = nullptr;
    free( featureIndexArray );
    featureIndexArray = nullptr;
}

TreeNode* TreeBuilder::Split(
    ValueIndexPair* valueIndexPairArr,
    unsigned int* featureIndexArray,
    const unsigned int* parentClassDist,
    const unsigned int numInstances,
    unsigned int height )
{
    // double giniImpurityMax = 0;
    double infoGainMax = 0;

    // The node is too small thus it is ignored.
    if (numInstances < MIN_NODE_SIZE) return nullptr;

    // The node is small, make it a leaf node.
    if (numInstances < MIN_NODE_SIZE_TO_SPLIT)
    {
        TreeNode* leaf = new TreeNode;
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
        LabelNode( leaf, parentClassDist );

        return leaf;
    }

    unsigned int selectedFeaIndex;
    double selectedThreshold;

    unsigned int* selectedChildSizeArr =
        (unsigned int*) malloc( NUM_CHILDREN * sizeof( unsigned int ) );
    unsigned int* childSizeArr =
        (unsigned int*) malloc( NUM_CHILDREN * sizeof( unsigned int ) );

    // Init child class distribution vector
    unsigned int** classDistArr =
        (unsigned int**) malloc( NUM_CHILDREN * sizeof( unsigned int* ) );
    unsigned int** selectedClassDistArr =
        (unsigned int**) malloc( NUM_CHILDREN * sizeof( unsigned int* ) );
    for (unsigned int childId = 0; childId < NUM_CHILDREN; childId++)
    {
        classDistArr[childId] = ( unsigned int* )
            malloc( numClasses * sizeof( unsigned int ) );
        selectedClassDistArr[childId] = ( unsigned int* )
            malloc( numClasses * sizeof( unsigned int ) );
    }

    // Store sorted values of that feature with indices
    ValueIndexPair* selectedValueIndexPairArr =
        (ValueIndexPair*) malloc( numInstances * sizeof( ValueIndexPair ) );
    // Store split threshold candidates
    double* splitCandidates = (double*) malloc( numInstances * sizeof( double ) );

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
            valueIndexPairArr[i].featureValue =
                instanceTable[valueIndexPairArr[i].featureIndex].
                    featureAttrArray[randFeaIndex];
        qsort(
            valueIndexPairArr,
            numInstances,
            sizeof( ValueIndexPair ),
            Compare );

        for (unsigned int i = 0; i < numInstances; i++)
            splitCandidates[i] = valueIndexPairArr[i].featureValue;
        unsigned int numSplit =
            removeDuplicates( splitCandidates, numInstances ) - 1;

        // Reset child data and child class distribution
        unsigned int splitIndex = 0;
        for (unsigned int classId = 0; classId < numClasses; classId++)
            classDistArr[0][classId] = 0;
        memcpy(
            classDistArr[1],
            parentClassDist,
            numClasses * sizeof( unsigned int ) );

        bool featureIndexStored = false;

        // Find best split threshold
        for (unsigned int valueIndex = 0; valueIndex < numSplit; valueIndex++)
        {
            double splitPoint =
                (splitCandidates[valueIndex] + splitCandidates[valueIndex + 1]) / 2.0;

            while (splitIndex < numInstances)
            {
                const Instance& instance =
                    instanceTable[valueIndexPairArr[splitIndex].featureIndex];
                if (instance.featureAttrArray[randFeaIndex] >= splitPoint)
                    break;

                classDistArr[0][instance.classIndex]++;
                classDistArr[1][instance.classIndex]--;

                splitIndex++;
            }

            childSizeArr[0] = splitIndex;
            childSizeArr[1] = numInstances - splitIndex;

            // double giniImpurity = giniParent;
            double infoGain = entropyParent;
            
            // Compute entropy of children
            for (unsigned int childId = 0; childId < NUM_CHILDREN; childId++)
            {
                double numChildren = childSizeArr[childId];
                double entropyChild = ComputeEntropy(
                    classDistArr[childId],
                    numChildren );
                infoGain -= numChildren / (double) numInstances * entropyChild;
                // double giniChild = ComputeGini(
                //     classDistArr[childId],
                //     numChildren );
                // giniImpurity -= numChildren / (double) numInstances * giniChild;
            }

            // Get max split criteria and related feature
            // if (giniImpurityMax < giniImpurity)
            if (infoGainMax < infoGain)
            {
                if (!featureIndexStored)
                {
                    memcpy(
                        selectedValueIndexPairArr,
                        valueIndexPairArr,
                        numInstances * sizeof( ValueIndexPair ) );

                    selectedFeaIndex = randFeaIndex;
                    featureIndexStored = true;
                }

                // Faster than memcpy for short arrays
                for (unsigned int childId = 0; childId < NUM_CHILDREN; childId++)
                {
                    selectedChildSizeArr[childId] = childSizeArr[childId];
                    for (unsigned int classId = 0; classId < numClasses; classId++)
                        selectedClassDistArr[childId][classId] =
                            classDistArr[childId][classId];
                }

                // giniImpurityMax = giniImpurity;
                infoGainMax = infoGain;
                selectedThreshold = splitPoint;

                if (!gainFound) gainFound = true;
            }
        }
    }

    free( childSizeArr );
    childSizeArr = nullptr;
    free( splitCandidates );
    splitCandidates = nullptr;

    for (unsigned int childId = 0; childId < NUM_CHILDREN; childId++)
        free( classDistArr[childId] );
    free( classDistArr );
    classDistArr = nullptr;
    //printf( "\n----------------------------------------\n");
    //printf( "Height: %d\n", height );

    TreeNode* node;

    // Split threshold not found, 
    // or gini impurity / info gain exceeds threshold,
    // thus have reached leaf node.
    if (!gainFound) node = nullptr;
    // Split node
    else
    {
        //printf( "Feature selected: %s\n", featureVec[selectedFeaIndex].name );
        //printf( "Gini of parent: %f\n", giniParent );
        //printf( "Max Gini split get: %f\n", giniImpurityMax );
        
        node = new TreeNode;
        node->featureIndex = selectedFeaIndex;
        node->threshold    = selectedThreshold;
        node->labeled      = false;

        height++;

        bool emptyChildFound = false;

        // Split children
        for (unsigned int childId = 0; childId < NUM_CHILDREN; childId++)
        {
            ValueIndexPair* childValueIndexPairArr = (ValueIndexPair*)
                malloc( selectedChildSizeArr[childId] * sizeof( ValueIndexPair ) );
            // Consider NUM_CHILDREN is 2, childId is either 0 or 1.
            ValueIndexPair* offset = selectedValueIndexPairArr + ((childId) ?
                selectedChildSizeArr[0] : 0);
            memcpy(
                childValueIndexPairArr,
                offset,
                selectedChildSizeArr[childId] * sizeof( ValueIndexPair ) );
            
            TreeNode* childNode = Split(
                childValueIndexPairArr,
                featureIndexArray,
                selectedClassDistArr[childId],
                selectedChildSizeArr[childId],
                height );

            free( selectedClassDistArr[childId] );
            selectedClassDistArr[childId] = nullptr;
            free( childValueIndexPairArr );
            childValueIndexPairArr = nullptr;

            if (childNode == nullptr) emptyChildFound = true;
            node->childrenVec.push_back( childNode );
        }

        if (emptyChildFound) LabelNode( node, parentClassDist );
    }

    free( selectedClassDistArr );
    selectedClassDistArr = nullptr;
    free( selectedChildSizeArr );
    selectedChildSizeArr = nullptr;
    free( selectedValueIndexPairArr );
    selectedValueIndexPairArr = nullptr;

    return node;
}

void TreeBuilder::PrintTree( const TreeNode* iter, unsigned int h )
{
    if (iter == nullptr || iter->labeled || h > 3) return;

    for (unsigned int i = 0; i <= h; i++) printf( "-" );
    printf( "%s, ", featureVec[iter->featureIndex].name );
    printf( "%f\n", iter->threshold );

    for (const TreeNode* child : iter->childrenVec)
        PrintTree( child, h + 1 );
}

TreeNode* TreeBuilder::GetRoot()
{
    return root;
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
    node->labeled = true;
}

void TreeBuilder::DestroyNode( TreeNode* node )
{
    if (node == nullptr) return;

    if (!node->childrenVec.empty())
    {
        for (TreeNode* child : node->childrenVec) DestroyNode( child );
        node->childrenVec.clear();
    }

    delete node;
    node = nullptr;
}
