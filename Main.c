
#include "ArffImporter.h"
#include "Classifier.h"


int main()
{
    ArffImporter trainSetImporter;
    trainSetImporter.Read( "Dataset/train/train-first50.arff" );

    ArffImporter testSetImporter;
    testSetImporter.Read( "Dataset/test/dev-first50.arff" );

    Classifier classifier;
    classifier.Train(
        trainSetImporter.GetInstances(), 
        trainSetImporter.GetFeatures(), 
        trainSetImporter.GetClassAttr() );
    classifier.Classify( testSetImporter.GetInstances() );
    classifier.Analyze(
        "This is bad",
        trainSetImporter.GetFeatures(),
        trainSetImporter.GetClassAttr() );

    return 0;
}
