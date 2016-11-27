
#include "ArffImporter.h"
#include "Classifier.h"


extern "C"
{
    void trainAndTest( int numTrees )
    {
        ArffImporter trainSetImporter;
        trainSetImporter.Read( "Dataset/train/train-first1000.arff" );

        ArffImporter testSetImporter;
        testSetImporter.Read( "Dataset/test/dev-first1000.arff" );

        Classifier classifier;
        classifier.Train(
            trainSetImporter.GetItems(), 
            trainSetImporter.GetFeatures(), 
            trainSetImporter.GetClassAttr() );
        classifier.Classify( testSetImporter.GetItems() );
    }
}

