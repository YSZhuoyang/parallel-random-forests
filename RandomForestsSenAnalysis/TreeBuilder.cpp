
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
    //double giniSplitMax = 0;
    double infoGainMax = 0;
    unsigned int numInstances = iiv.size();

    // The node is too small thus it is ignored.
    if (numInstances < MIN_NODE_SIZE)
    {
        return nullptr;
    }
    // The node is small, make it a leaf node.
    else if (numInstances < MIN_NODE_SIZE_TO_SPLIT)
    {
        TreeNode* leaf = new TreeNode;
        LabelNode( leaf, iiv );
        
        return leaf;
    }

    //double giniParent = ComputeGini( iiv );
    // Compute entropy of this node.
    double entropyParent = ComputeEntropy( iiv );

    // All node is pure.
    //if (giniParent <= 0.0)
    if (entropyParent <= 0.0)
    {
        TreeNode* leaf = new TreeNode;
        LabelNode( leaf, iiv );

        return leaf;
    }

    unsigned int selectedFeaIndex;
    double selectedThreshold;
    vector<vector<unsigned int>> selectedChildrenIndex;

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

        vector<vector<unsigned int>> groups;
        groups.resize( NUM_CHILDREN );
        groups[1] = iiv;

        // Find split threshold
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
                    swap( groups[1][i], groups[1].back() );
                    groups[1].pop_back();

                    groupSize--;
                }
                else i++;
            }

            //double giniSplit = giniParent;
            double infoGain = entropyParent;
            
            // Compute entropy of children
            for (const vector<unsigned int>& group : groups)
            {
                //double giniChild = ComputeGini( group );
                double entropyChild = ComputeEntropy( group );
                double numChildren = group.size();
                //giniSplit -= numChildren / (double) numInstances * giniChild;
                infoGain -= numChildren / (double) numInstances * entropyChild;
            }

            // Get max info gain and related feature
            //if (giniSplitMax < giniSplit)
            if (infoGainMax < infoGain)
            {
                //giniSplitMax = giniSplit;
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

    //printf( "\n----------------------------------------\n");
    //printf( "Height: %d\n", height );

    TreeNode* node = new TreeNode;

    // Split threshold not found, 
    // or gini split / info gain exceeds threshold,
    // thus have reached leaf node.
    if (!gainFound) return nullptr;
    // Split node
    else
    {
        //printf( "Feature selected: %s\n", featureVec[selectedFeaIndex].name );
        //printf( "Gini of parent: %f\n", giniParent );
        //printf( "Max Gini split get: %f\n", giniSplitMax );
        
        node->featureIndex = selectedFeaIndex;
        node->threshold    = selectedThreshold;
        //node->gini         = giniParent;
        //node->giniSplit    = giniSplitMax;
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

        if (emptyChildFound) LabelNode( node, iiv );
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

double TreeBuilder::ComputeEntropy( const vector<unsigned int>& iiv )
{
    unsigned int totalInstCount = iiv.size();
    if (totalInstCount == 0) return 0.0;

    unsigned int* bucketArray = GetDistribution( iiv );
    double entropy = 0.0;

    for (unsigned short i = 0; i < numClasses; i++)
    {
        if (bucketArray[i] > 0 && bucketArray[i] < totalInstCount)
        {
            double temp = (double) bucketArray[i] / totalInstCount;
            entropy -= temp * log2( temp );
        }
    }

    free( bucketArray );
    bucketArray = nullptr;

    return entropy;
}

double TreeBuilder::ComputeGini( const vector<unsigned int>& iiv )
{
    double totalInstCount = iiv.size();
    if (totalInstCount == 0) return 0.0;

    unsigned int* bucketArray = GetDistribution( iiv );
    double gini = 1.0;

    for (unsigned short i = 0; i < numClasses; i++)
    {
        double temp = (double) bucketArray[i] / totalInstCount;
        gini -= temp * temp;
    }

    free( bucketArray );
    bucketArray = nullptr;

    return gini;
}

unsigned int* TreeBuilder::GetDistribution(
    const vector<unsigned int>& iiv )
{
    unsigned int* bucketArray = 
        (unsigned int*) calloc( numClasses, sizeof( unsigned int ) );
    for (const unsigned int& instIndex : iiv)
        bucketArray[instanceVec[instIndex].classIndex]++;

    return bucketArray;
}

void TreeBuilder::LabelNode(
    TreeNode* node,
    const vector<unsigned int>& iiv )
{
    if (node == nullptr) return;

    unsigned int* bucketArray = GetDistribution( iiv );
    // Select the class of the largest class group.
    node->classIndex = getIndexOfMax( bucketArray, numClasses );
    node->labeled = true;

    free( bucketArray );
    bucketArray = nullptr;
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