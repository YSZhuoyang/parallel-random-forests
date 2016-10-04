
#include "ArffImporter.h"
#include "Classifier.h"


int main()
{
    ArffImporter trainSetImporter;
    trainSetImporter.Read( "Dataset/train/train-first1000.arff" );

    ArffImporter testSetImporter;
    testSetImporter.Read( "Dataset/test/dev-first1000.arff" );

    Classifier classifier(
        trainSetImporter.GetFeatures(), 
        trainSetImporter.GetClassAttr() );
    classifier.Train( trainSetImporter.GetItems() );
    classifier.Classify( testSetImporter.GetItems() );

    return 0;
}
