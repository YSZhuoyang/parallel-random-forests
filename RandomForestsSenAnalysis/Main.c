
#include "ArffImporter.h"
#include "Classifier.h"


extern "C"
{
    void Train(
        unsigned int numTrees, 
        unsigned int numFeaPerTree )
    {
        ArffImporter trainSetImporter;
        trainSetImporter.Read(
            "RandomForestsSenAnalysis/Dataset/train/train-first50.arff" );

        Classifier classifier;
        classifier.Configure( numTrees, numFeaPerTree );
        classifier.Train(
            trainSetImporter.GetItems(),
            trainSetImporter.GetFeatures(),
            trainSetImporter.GetClassAttr() );
    }

    float Test()
    {
        ArffImporter testSetImporter;
        testSetImporter.Read( 
            "RandomForestsSenAnalysis/Dataset/test/dev-first50.arff" );

        Classifier classifier;
        return classifier.Classify(
            testSetImporter.GetItems(), 
            testSetImporter.GetClassAttr() );
    }

    char* Analyze( char* sentence )
    {
        ArffImporter trainSetImporter;
        trainSetImporter.Read(
            "RandomForestsSenAnalysis/Dataset/train/train-first50.arff" );

        Classifier classifier;
        char* label = classifier.Analyze(
            sentence,
            trainSetImporter.GetFeatures(),
            trainSetImporter.GetClassAttr() );    
        printf( "Labeled with: %s\n", label );

        return label;
    }

    void FreeMemo( char* str )
    {
        free( str );
        str = nullptr;
    }
}

