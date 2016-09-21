

#include "ArffImporter.h"
#include "TreeBuilder.h"


int main()
{
    ArffImporter importer;
    importer.Read( "Dataset/train/train-first10.arff" );

    TreeBuilder treeBuilder;
    treeBuilder.BuildTree( importer.GetItems(), importer.GetFeatures(), importer.GetClassAttr() );
    
    return 0;
}



