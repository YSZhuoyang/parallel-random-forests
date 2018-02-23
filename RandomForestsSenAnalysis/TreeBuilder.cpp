
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
    
    root = Split(
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

    // The node is small, make it a leaf node.
    if (numInstances < MIN_NODE_SIZE_TO_SPLIT)
    {
        TreeNode* leaf = new TreeNode;
        leaf->classIndex = getIndexOfMax( parentClassDist, numClasses );
        
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
        leaf->classIndex = getIndexOfMax( parentClassDist, numClasses );

        return leaf;
    }

    unsigned int selectedFeaIndex;
    double selectedThreshold;

    unsigned int childSizeArr[NUM_CHILDREN];
    unsigned int selectedChildSizeArr[NUM_CHILDREN];

    // Init child class distribution vector
    unsigned int* classDistArr[NUM_CHILDREN];
    unsigned int* selectedClassDistArr[NUM_CHILDREN];
    for (unsigned int childId = 0; childId < NUM_CHILDREN; childId++)
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
            []( const MiniInstance& eleX, 
                const MiniInstance& eleY )
            {
                return eleX.featureValue < eleY.featureValue;
            } );

        // Reset split index and child class distribution
        memset( classDistArr[0], 0, numClasses * sizeof( unsigned int ) );
        memmove(
            classDistArr[1],
            parentClassDist,
            numClasses * sizeof( unsigned int ) );

        bool featureIndexStored = false;
        unsigned int numCandidates = numInstances - 1;

        // Find the best split threshold
        for (unsigned int candidateId = 0;
            candidateId < numCandidates; candidateId++)
        {
            unsigned int nextCandidateId = candidateId + 1;
            
            classDistArr[0][valueIndexTupleArr[candidateId].classIndex]++;
            classDistArr[1][valueIndexTupleArr[candidateId].classIndex]--;

            if (valueIndexTupleArr[candidateId].featureValue <
                valueIndexTupleArr[nextCandidateId].featureValue)
            {
                double splitThreshold =
                    (valueIndexTupleArr[candidateId].featureValue + 
                    valueIndexTupleArr[nextCandidateId].featureValue) / 2.0;
                childSizeArr[0] = nextCandidateId;
                childSizeArr[1] = numInstances - nextCandidateId;

                // double giniImpurity = giniParent;
                double infoGain = entropyParent;
                
                // Compute entropy of children
                for (unsigned int childId = 0; childId < NUM_CHILDREN; childId++)
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

                    for (unsigned int childId = 0; childId < NUM_CHILDREN; childId++)
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

    for (unsigned int childId = 0; childId < NUM_CHILDREN; childId++)
        free( classDistArr[childId] );
    //printf( "\n----------------------------------------\n");
    //printf( "Height: %d\n", height );

    TreeNode* node;

    // Split threshold not found, 
    // or gini impurity / info gain exceeds threshold,
    // thus have reached leaf node.
    if (!gainFound)
    {
        for (unsigned int childId = 0; childId < NUM_CHILDREN; childId++)
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

        height++;

        bool emptyChildFound = false;

        // Split children
        for (unsigned int childId = 0; childId < NUM_CHILDREN; childId++)
        {
            // Consider NUM_CHILDREN is 2, childId is either 0 or 1.
            ValueIndexTuple* childValueIndexTupleArr =
                selectedValueIndexTupleArr + childId * selectedChildSizeArr[0];
            
            TreeNode* childNode = Split(
                childValueIndexTupleArr,
                featureIndexArray,
                selectedClassDistArr[childId],
                selectedChildSizeArr[childId],
                height );

            free( selectedClassDistArr[childId] );
            selectedClassDistArr[childId] = nullptr;

            if (childNode == nullptr) emptyChildFound = true;
            node->childrenVec.push_back( childNode );
        }

        if (emptyChildFound)
            node->classIndex = getIndexOfMax( parentClassDist, numClasses );
    }

    free( selectedValueIndexTupleArr );
    selectedValueIndexTupleArr = nullptr;

    return node;
}

void TreeBuilder::PrintTree( const TreeNode* iter, unsigned int h )
{
    if (iter == nullptr || iter->childrenVec.empty()) return;

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
