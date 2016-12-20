
#include "ArffImporter.h"
#include "Classifier.h"


int main()
{
    ArffImporter trainSetImporter;
    trainSetImporter.Read( "Dataset/train/train-first1000.arff" );

    ArffImporter testSetImporter;
    testSetImporter.Read( "Dataset/test/dev-first1000.arff" );

    Classifier classifier;
    classifier.Train(
        trainSetImporter.GetInstances(), 
        trainSetImporter.GetFeatures(), 
        trainSetImporter.GetClassAttr() );
    classifier.Classify( testSetImporter.GetInstances() );

    return 0;
}
