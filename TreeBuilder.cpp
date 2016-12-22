
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

    vector<unsigned int> instIndexVec;
    instIndexVec.reserve( numInstances );
    for (unsigned int i = 0; i < numInstances; i++)
        instIndexVec.push_back( i );
    
    root = Split( instIndexVec, featureIndexArray, 0 );

    free( featureIndexArray );
    featureIndexArray = nullptr;
}

TreeNode* TreeBuilder::Split(
    const vector<unsigned int>& iiv,
    unsigned int* featureIndexArray,
    unsigned int height )
{
    // double giniImpurityMax = 0;
    double infoGainMax = 0;
    unsigned int numInstances = iiv.size();

    // The node is too small thus it is ignored.
    if (numInstances < MIN_NODE_SIZE) return nullptr;

    // Compute entropy of parent node.
    unsigned int* parentClassDist = GetDistribution( iiv );

    // The node is small, make it a leaf node.
    if (numInstances < MIN_NODE_SIZE_TO_SPLIT)
    {
        TreeNode* leaf = new TreeNode;
        LabelNode( leaf, parentClassDist );
        
        return leaf;
    }

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
    vector<vector<unsigned int>> selectedChildrenIndex;

    // Init child class distribution and instance vector
    vector<vector<unsigned int>> groups;
    groups.resize( NUM_CHILDREN );
    groups[1] = iiv;

    vector<unsigned int*> classDistVec;
    classDistVec.resize( NUM_CHILDREN );
    for (unsigned int childId = 0; childId < NUM_CHILDREN; childId++)
        classDistVec[childId] = ( unsigned int* )
            calloc( numClasses, sizeof( unsigned int ) );

    unsigned int numRestFeaToSelect = numFeaturesToSelect;
    unsigned int numRestFea = numFeaturesTotal;
    bool gainFound = false;

    // Find the best split feature and threshold
    while ((numRestFeaToSelect-- > 0 || !gainFound) && numRestFea > 0)
    {
        // Sample (note max of rand() is around 24000)
        unsigned int randPos = rand() % numRestFea;
        unsigned int randFeaIndex = featureIndexArray[randPos];
        // Swap
        featureIndexArray[randPos] = featureIndexArray[--numRestFea];
        featureIndexArray[numRestFea] = randFeaIndex;

        // Get all values of that feature and sort them.
        double* valueArr = (double*) malloc( numInstances * sizeof( double ) );
        for (unsigned int i = 0; i < numInstances; i++)
            valueArr[i] = instanceVec[iiv[i]].featureAttrArray[randFeaIndex];
        qsort( valueArr, numInstances, sizeof( double ), Compare );
        unsigned int numSplit = removeDuplicates( valueArr, numInstances ) - 1;

        // Reset child data and child class distribution
        groups[0].clear();
        groups[1] = iiv;

        for (unsigned int classId = 0; classId < numClasses; classId++)
            classDistVec[0][classId] = 0;
        memcpy(
            classDistVec[1],
            parentClassDist,
            numClasses * sizeof( unsigned int ) );

        // Find best split threshold
        for (unsigned int valueIndex = 0; valueIndex < numSplit; valueIndex++)
        {
            unsigned int groupSize = groups[1].size();
            unsigned int i = 0;
            double splitPoint =
                (valueArr[valueIndex] + valueArr[valueIndex + 1]) / 2.0;
            
            while (groupSize > i)
            {
                const Instance& instance = instanceVec[groups[1][i]];

                if (instance.featureAttrArray[randFeaIndex] < splitPoint)
                {
                    groups[0].push_back( groups[1][i] );
                    groups[1][i] = groups[1].back();
                    groups[1].pop_back();

                    classDistVec[0][instance.classIndex]++;
                    classDistVec[1][instance.classIndex]--;

                    groupSize--;
                }
                else i++;
            }

            // double giniImpurity = giniParent;
            double infoGain = entropyParent;
            
            // Compute entropy of children
            for (unsigned int childId = 0; childId < NUM_CHILDREN; childId++)
            {
                double numChildren = groups[childId].size();
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
                // giniImpurityMax = giniImpurity;
                infoGainMax = infoGain;
                selectedChildrenIndex = groups;
                selectedThreshold = splitPoint;
                selectedFeaIndex = randFeaIndex;
                gainFound = true;
            }
        }

        free( valueArr );
        valueArr = nullptr;
    }

    for (unsigned int classId = 0; classId < numClasses; classId++)
        free( classDistVec[classId] );

    free( parentClassDist );
    parentClassDist = nullptr;

    //printf( "\n----------------------------------------\n");
    //printf( "Height: %d\n", height );

    TreeNode* node = new TreeNode;

    // Split threshold not found, 
    // or gini impurity / info gain exceeds threshold,
    // thus have reached leaf node.
    if (!gainFound) return nullptr;
    // Split node
    else
    {
        //printf( "Feature selected: %s\n", featureVec[selectedFeaIndex].name );
        //printf( "Gini of parent: %f\n", giniParent );
        //printf( "Max Gini split get: %f\n", giniImpurityMax );
        
        node->featureIndex = selectedFeaIndex;
        node->threshold    = selectedThreshold;
        node->labeled      = false;

        height++;

        bool emptyChildFound = false;

        // Split children
        for (const vector<unsigned int>& childGroup : selectedChildrenIndex)
        {
            TreeNode* childNode = Split( childGroup, 
                featureIndexArray, height );
            if (childNode == nullptr) emptyChildFound = true;
            node->childrenVec.push_back( childNode );
        }

        if (emptyChildFound) LabelNode( node, parentClassDist );
    }

    return node;
}

void TreeBuilder::PrintTree( const TreeNode* iter, unsigned int h )
{
    if (iter == nullptr || iter->labeled) return;

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
    const vector<unsigned int>& iiv )
{
    unsigned int* bucketArray = 
        (unsigned int*) calloc( numClasses, sizeof( unsigned int ) );
    for (const unsigned int& instIndex : iiv)
        bucketArray[instanceVec[instIndex].classIndex]++;

    return bucketArray;
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
