
#include "ArffImporter.h"
#include "Classifier.h"


int main()
{
    ArffImporter trainSetImporter;
    trainSetImporter.Read( "Dataset/train/train-first1000.arff" );

    ArffImporter testSetImporter;
    testSetImporter.Read( "Dataset/train/train-first1000.arff" );

    Classifier classifier;
    classifier.Train(
        trainSetImporter.GetItems(), 
        trainSetImporter.GetFeatures(), 
        trainSetImporter.GetClassAttr() );
    classifier.Classify( testSetImporter.GetItems() );
    classifier.Analyze(
        "bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad",
        trainSetImporter.GetFeatures(),
        trainSetImporter.GetClassAttr() );

    return 0;
}
