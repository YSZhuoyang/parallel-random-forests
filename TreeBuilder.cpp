
#include "TreeBuilder.h"
#include <bits/stdc++.h>


TreeBuilder::TreeBuilder()
{
    
}

TreeBuilder::~TreeBuilder()
{
    
}

void TreeBuilder::Init(
    const vector<NumericAttr>& fv,
    const vector<char*>& cv )
{
    featureVec = fv;
    classVec = cv;
    numFeaturesTotal = featureVec.size();
    numClasses = classVec.size();
}

void TreeBuilder::BuildTree(
    double** instanceTable,
    const unsigned int numInstances,
    const unsigned int numFeaToSelect )
{
    numFeaturesToSelect = numFeaToSelect;
    
    unsigned int* featureIndexArray = 
        (unsigned int*) malloc( numFeaturesTotal * sizeof( unsigned int ) );
    for (unsigned int i = 0; i < numFeaturesTotal; i++)
        featureIndexArray[i] = i;

    // ValueIndexPair* valueIndexPairArr =
    //     (ValueIndexPair*) malloc( numInstancesTotal * sizeof( ValueIndexPair ) );
    unsigned int* initialClassDist = 
        (unsigned int*) calloc( numClasses, sizeof( unsigned int ) );
    for (unsigned int i = 0; i < numInstances; i++)
    {
        // Get overall distribution
        initialClassDist[(unsigned int) instanceTable[i][numFeaturesTotal]]++;
        // Init data indices
        // valueIndexPairArr[i].featureIndex = i;
    }

    // printf( "c1: %u, c2: %u\n", initialClassDist[0], initialClassDist[1] );
    
    root = Split(
        instanceTable,
        featureIndexArray,
        initialClassDist,
        numInstances,
        0 );

    free( initialClassDist );
    initialClassDist = nullptr;
    // free( valueIndexPairArr );
    // valueIndexPairArr = nullptr;
    free( featureIndexArray );
    featureIndexArray = nullptr;
}

TreeNode* TreeBuilder::Split(
    double** instanceTable,
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
    double** selectedInstanceTable =
        (double**) malloc( numInstances * sizeof( double* ) );
    // ValueIndexPair* selectedValueIndexPairArr =
    //     (ValueIndexPair*) malloc( numInstances * sizeof( ValueIndexPair ) );

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
        // for (unsigned int i = 0; i < numInstances; i++)
        //     instanceTable[i].splitCanFeaId = randFeaIndex;
            // valueIndexPairArr[i].featureValue =
            //     instanceTable[valueIndexPairArr[i].featureIndex].
            //         featureAttrArray[randFeaIndex];
        // for (unsigned int i = 0; i < numInstances; i++)
        //     instanceTable[i].splitCanFeaId = randFeaIndex;
        Comp comp( randFeaIndex );
        sort(
            instanceTable,//valueIndexPairArr,
            instanceTable + numInstances,
            comp );

        // if (height == 0)
        // {
        //     for (int i = 0; i < 50; i++)
        //         printf( "%f ", instanceTable[i][randFeaIndex] );
        //     printf( "\n after:\n" );
        // }

        // qsort_r(
        //     instanceTable,
        //     numInstances,
        //     sizeof( double* ),
        //     Compare,
        //     &randFeaIndex );

        // if (height == 0)
        // {
        //     for (int i = 0; i < 50; i++)
        //         printf( "%f ", instanceTable[i][randFeaIndex] );
        //     printf( "\n" );
        // }

        // Reset child class distribution
        unsigned int splitIndex = 0;
        for (unsigned int classId = 0; classId < numClasses; classId++)
            classDistArr[0][classId] = 0;
        memcpy(
            classDistArr[1],
            parentClassDist,
            numClasses * sizeof( unsigned int ) );

        bool sortedInstTableStored = false;
        double priorCandidate = 
            instanceTable[0][randFeaIndex];
        // double priorCandidate = valueIndexPairArr[0].featureValue;

        // Find the best split threshold
        for (unsigned int candidateId = 1;
            candidateId < numInstances; candidateId++)
        {
            // if (priorCandidate == valueIndexPairArr[candidateId].featureValue)
            if (priorCandidate == 
                instanceTable[candidateId][randFeaIndex])
                continue;
            
            double splitThreshold = (priorCandidate + 
                instanceTable[candidateId][randFeaIndex]) / 2.0;
            priorCandidate =
                instanceTable[candidateId][randFeaIndex];

            while (splitIndex < numInstances)
            {
                const double* instance = instanceTable[splitIndex];
                if (instance[randFeaIndex] >= splitThreshold)
                    break;

                classDistArr[0][(unsigned int) instance[numFeaturesTotal]]++;
                classDistArr[1][(unsigned int) instance[numFeaturesTotal]]--;

                splitIndex++;
            }

            childSizeArr[0] = splitIndex;
            childSizeArr[1] = numInstances - splitIndex;

            // if (height == 0) printf( "c1: %u, c2: %u\n", childSizeArr[0], childSizeArr[1] );

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

            // Get max split outcome and related feature
            // if (giniImpurityMax < giniImpurity)
            if (infoGainMax < infoGain)
            {
                if (!sortedInstTableStored)
                {
                    memcpy(
                        selectedInstanceTable,
                        instanceTable,
                        numInstances * sizeof( double* ) );

                    selectedFeaIndex = randFeaIndex;
                    sortedInstTableStored = true;
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
                selectedThreshold = splitThreshold;

                if (!gainFound) gainFound = true;
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
            double** childInstanceTable = (double**)
                malloc( selectedChildSizeArr[childId] * sizeof( double* ) );
            // ValueIndexPair* childValueIndexPairArr = (ValueIndexPair*)
            //     malloc( selectedChildSizeArr[childId] * sizeof( ValueIndexPair ) );
            // Consider NUM_CHILDREN is 2, childId is either 0 or 1.
            double** offset = selectedInstanceTable + ((childId) ?
                selectedChildSizeArr[0] : 0);
            memcpy(
                childInstanceTable,
                offset,
                selectedChildSizeArr[childId] * sizeof( double* ) );
            
            TreeNode* childNode = Split(
                childInstanceTable,
                featureIndexArray,
                selectedClassDistArr[childId],
                selectedChildSizeArr[childId],
                height );

            free( selectedClassDistArr[childId] );
            selectedClassDistArr[childId] = nullptr;
            free( childInstanceTable );
            childInstanceTable = nullptr;

            if (childNode == nullptr) emptyChildFound = true;
            node->childrenVec.push_back( childNode );
        }

        if (emptyChildFound) LabelNode( node, parentClassDist );
    }

    free( selectedInstanceTable );
    selectedInstanceTable = nullptr;

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
