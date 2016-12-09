
#include "ArffImporter.h"
#include "Classifier.h"


int main()
{
    ArffImporter trainSetImporter;
    trainSetImporter.Read( "Dataset/train/train-first10.arff" );

    ArffImporter testSetImporter;
    testSetImporter.Read( "Dataset/train/train-first10.arff" );

    Classifier classifier;
    classifier.Train(
        trainSetImporter.GetItems(), 
        trainSetImporter.GetFeatures(), 
        trainSetImporter.GetClassAttr() );
    classifier.Classify( testSetImporter.GetItems() );
    classifier.Analyze(
        "This is bad",
        trainSetImporter.GetFeatures(),
        trainSetImporter.GetClassAttr() );

    return 0;
}
