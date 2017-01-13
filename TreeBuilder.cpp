
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
    const vector<Instance>& iv )
{
    featureVec = fv;
    classVec = cv;
    instanceVec = iv;
    numFeaturesTotal = fv.size();
    numClasses = classVec.size();
}

void TreeBuilder::BuildTree( const unsigned int numFeaToSelect )
{
    numFeaturesToSelect = numFeaToSelect;
    unsigned int numInstances = instanceVec.size();
    
    unsigned int* featureIndexArray = 
        (unsigned int*) malloc( numFeaturesTotal * sizeof( unsigned int ) );
    for (unsigned int i = 0; i < numFeaturesTotal; i++)
        featureIndexArray[i] = i;

    unsigned int* instIndexArr =
        (unsigned int*) malloc( numInstances * sizeof( unsigned int ) );
    for (unsigned int i = 0; i < numInstances; i++)
        instIndexArr[i] = i;
    
    root = Split( instIndexArr, featureIndexArray, numInstances, 0 );

    free( instIndexArr );
    instIndexArr = nullptr;
    free( featureIndexArray );
    featureIndexArray = nullptr;
}

TreeNode* TreeBuilder::Split(
    unsigned int* iia,
    unsigned int* featureIndexArray,
    const unsigned int numInstances,
    unsigned int height )
{
    // double giniImpurityMax = 0;
    double infoGainMax = 0;

    // The node is too small thus it is ignored.
    if (numInstances < MIN_NODE_SIZE) return nullptr;

    unsigned int* parentClassDist = GetDistribution( iia, numInstances );

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

    // Init child class distribution and instance vector
    vector<unsigned int> selectedChildSizeVec;
    selectedChildSizeVec.resize( NUM_CHILDREN );

    vector<unsigned int*> classDistVec;
    classDistVec.resize( NUM_CHILDREN );
    for (unsigned int childId = 0; childId < NUM_CHILDREN; childId++)
        classDistVec[childId] = ( unsigned int* )
            malloc( numClasses * sizeof( unsigned int ) );

    // Store sorted index sequence
    unsigned int* selectedInstIndicesArr =
        (unsigned int*) malloc( numInstances * sizeof( unsigned int ) );
    // Store all values of that feature with indices
    ValueIndexPair* valueIndexPairArr =
        (ValueIndexPair*) malloc( numInstances * sizeof( ValueIndexPair ) );
    double* valueArr = (double*) malloc( numInstances * sizeof( double ) );
    
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
        {
            valueIndexPairArr[i].featureValue =
                instanceVec[iia[i]].featureAttrArray[randFeaIndex];
            valueIndexPairArr[i].featureIndex = iia[i];
        }
        qsort( valueIndexPairArr, numInstances, sizeof( ValueIndexPair ), Compare );

        for (unsigned int i = 0; i < numInstances; i++)
        {
            valueArr[i] = valueIndexPairArr[i].featureValue;
            iia[i] = valueIndexPairArr[i].featureIndex;
        }
        unsigned int numSplit = removeDuplicates( valueArr, numInstances ) - 1;

        // Reset child data and child class distribution
        unsigned int splitIndex = 0;
        for (unsigned int classId = 0; classId < numClasses; classId++)
            classDistVec[0][classId] = 0;
        memcpy(
            classDistVec[1],
            parentClassDist,
            numClasses * sizeof( unsigned int ) );

        vector<unsigned int> childSizeVec;
        childSizeVec.resize( NUM_CHILDREN );

        // Find best split threshold
        for (unsigned int valueIndex = 0; valueIndex < numSplit; valueIndex++)
        {
            double splitPoint =
                (valueArr[valueIndex] + valueArr[valueIndex + 1]) / 2.0;
            
            while (splitIndex < numInstances)
            {
                const Instance& instance = instanceVec[iia[splitIndex]];
                if (instance.featureAttrArray[randFeaIndex] >= splitPoint)
                    break;
                
                classDistVec[0][instance.classIndex]++;
                classDistVec[1][instance.classIndex]--;

                splitIndex++;
            }

            childSizeVec[0] = splitIndex;
            childSizeVec[1] = numInstances - splitIndex;

            // double giniImpurity = giniParent;
            double infoGain = entropyParent;
            
            // Compute entropy of children
            for (unsigned int childId = 0; childId < NUM_CHILDREN; childId++)
            {
                double numChildren = childSizeVec[childId];
                double entropyChild = ComputeEntropy(
                    classDistVec[childId],
                    numChildren );
                infoGain -= numChildren / (double) numInstances * entropyChild;
                // double giniChild = ComputeGini(
                //     classDistVec[childId],
                //     numChildren );
                // giniImpurity -= numChildren / (double) numInstances * giniChild;
            }

            // Get max split criteria and related feature
            // if (giniImpurityMax < giniImpurity)
            if (infoGainMax < infoGain)
            {
                memcpy(
                    selectedInstIndicesArr,
                    iia,
                    numInstances * sizeof( unsigned int ) );
                
                // giniImpurityMax = giniImpurity;
                infoGainMax = infoGain;
                selectedChildSizeVec = childSizeVec;
                selectedThreshold = splitPoint;
                selectedFeaIndex = randFeaIndex;
                gainFound = true;
            }
        }
    }

    free( valueIndexPairArr );
    valueIndexPairArr = nullptr;
    free( valueArr );
    valueArr = nullptr;

    for (unsigned int childId = 0; childId < NUM_CHILDREN; childId++)
        free( classDistVec[childId] );

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
            unsigned int* childIndexArr = (unsigned int*)
                malloc( selectedChildSizeVec[childId] * sizeof( unsigned int ) );
            // Consider NUM_CHILDREN is always 2, childId is either 0 or 1.
            unsigned int* offset = selectedInstIndicesArr + ((childId) ?
                selectedChildSizeVec[0] : 0);
            memcpy(
                childIndexArr,
                offset,
                selectedChildSizeVec[childId] * sizeof( unsigned int ) );
            
            TreeNode* childNode = Split(
                childIndexArr,
                featureIndexArray,
                selectedChildSizeVec[childId],
                height );

            free( childIndexArr );
            childIndexArr = nullptr;

            if (childNode == nullptr) emptyChildFound = true;
            node->childrenVec.push_back( childNode );
        }

        if (emptyChildFound) LabelNode( node, parentClassDist );
    }

    free( selectedInstIndicesArr );
    selectedInstIndicesArr = nullptr;
    free( parentClassDist );
    parentClassDist = nullptr;

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
    {
        if (classDistribution[i] > 0 && classDistribution[i] < numInstances)
        {
            double temp = (double) classDistribution[i] / numInstances;
            entropy -= temp * log2( temp );
        }
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

inline unsigned int* TreeBuilder::GetDistribution(
    const unsigned int* iia,
    const unsigned int numInstances )
{
    unsigned int* distribution = 
        (unsigned int*) calloc( numClasses, sizeof( unsigned int ) );
    for (unsigned int i = 0; i < numInstances; i++)
        distribution[instanceVec[iia[i]].classIndex]++;

    return distribution;
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
