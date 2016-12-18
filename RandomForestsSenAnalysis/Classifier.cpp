
#include "Classifier.h"


Classifier::Classifier()
{

}

Classifier::~Classifier()
{
    // Destory trees
    for (TreeNode* root : rootVec) treeBuilder.DestroyNode( root );
    rootVec.clear();
}


void Classifier::Configure(
    unsigned int numTrees,
    unsigned int numFeaPerTree )
{
    this->numTrees = numTrees;
    this->numFeaPerTree = numFeaPerTree;
}

void Classifier::Train(
    const vector<Item>& iv, 
    const vector<NumericAttr>& fv, 
    const vector<char*>& cv )
{
    classVec = cv;
    featureVec = fv;

    printf( "Num features: %lu\n", fv.size() );
    
    /******************** Init tree constructer ********************/
    // Build a number of trees each having the same number of features.
    rootVec.reserve( numTrees );
    treeBuilder.Init( fv, cv, numFeaPerTree );

    time_t start,end;
    double dif;
    time( &start );

    //#pragma omp parallel for schedule(dynamic)
    for (unsigned int treeIndex = 0; treeIndex < numTrees; treeIndex++)
    {
        treeBuilder.BuildTree( iv );
        //#pragma omp critical
        rootVec.push_back( treeBuilder.GetRoot() );
        //treeBuilder.PrintTree( treeBuilder.GetRoot(), 0 );
    }

    time( &end );
    dif = difftime( end, start );
    printf( "Build forests: time taken is %.2lf seconds.\n", dif );

    SaveModel();
}

char* Classifier::Analyze(
    const char* str,
    const vector<NumericAttr>& featureVec,
    const vector<char*>& cv )
{
    classVec = cv;
    Item item = Tokenize( str, featureVec );
    int classIndex = Classify( item );

    free( item.featureAttrArray );
    item.featureAttrArray = nullptr;

    // Need to allocate new memo considering that class vector
    // will be cleared and all memo used will be released.
    char* label = (char*) malloc( sizeof( char ) );
    memcpy( label, classVec[classIndex], sizeof( char ) );

    return label;
}

float Classifier::Test(
    const vector<Item>& iv, 
    const vector<char*>& cv )
{
    if (rootVec.empty())
    {
        LoadModel();
        //printf( "Please train the model first.\n" );
        //return;
    }

    unsigned int correctCounter = 0;
    unsigned int totalNumber = iv.size();

    printf( "Test set size: %u\n", totalNumber );

    classVec = cv;

    //#pragma omp parallel for reduction(+: correctCounter) schedule(dynamic)
    for (unsigned int i = 0; i < totalNumber; i++)
        if (Classify( iv[i] ) == iv[i].classIndex) correctCounter++;

    float correctRate = (float) correctCounter / (float) totalNumber;
    float incorrectRate = 1.0f - correctRate;

    printf( "Correct rate: %f\n", correctRate );
    printf( "Incorrect rate: %f\n", incorrectRate );

    return correctRate;
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

void Classifier::SaveModel()
{
    std::ofstream ofs( MODEL_FILE_PATH );
    
    if (ofs.good())
    {
        boost::archive::text_oarchive oa{ ofs };
        oa << rootVec;
    }
    else
    {
        printf( "Output stream is in an error state, saving model failed\n" );
    }

    ofs.flush();
    ofs.close();
}

void Classifier::LoadModel()
{
    if (!rootVec.empty())
    {
        printf( "Model loaded already!" );
        return;
    }

    std::ifstream ifs( MODEL_FILE_PATH );

    if (ifs.good())
    {
        boost::archive::text_iarchive ia{ ifs };
        ia >> rootVec;
    }
    else
    {
        printf( "Input stream is in an error state, loading model failed\n" );
    }

    ifs.close();
}
