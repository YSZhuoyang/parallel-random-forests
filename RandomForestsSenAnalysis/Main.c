
#include "ArffImporter.h"
#include "Classifier.h"


extern "C"
{
    void Train( int numTrees )
    {
        ArffImporter trainSetImporter;
        trainSetImporter.Read( "RandomForestsSenAnalysis/Dataset/train/train-first50.arff" );

        Classifier classifier;
        classifier.Train(
            trainSetImporter.GetItems(), 
            trainSetImporter.GetFeatures(), 
            trainSetImporter.GetClassAttr() );
    }

    void Test()
    {
        ArffImporter testSetImporter;
        testSetImporter.Read( "RandomForestsSenAnalysis/Dataset/test/dev-first50.arff" );

        Classifier classifier;
        classifier.Classify(
            testSetImporter.GetItems(), 
            testSetImporter.GetClassAttr() );
    }
}

