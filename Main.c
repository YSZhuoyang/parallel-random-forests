
#include "ArffImporter.h"
#include "Classifier.h"


int main()
{
    ArffImporter trainSetImporter;
    trainSetImporter.Read( "Dataset/train/train-first200.arff" );

    ArffImporter testSetImporter;
    testSetImporter.Read( "Dataset/test/dev-first200.arff" );

    Classifier classifier(
        trainSetImporter.GetFeatures(), 
        trainSetImporter.GetClassAttr() );
    classifier.Train( trainSetImporter.GetItems() );
    classifier.Classify( testSetImporter.GetItems() );

    return 0;
}
