

#include "ArffImporter.h"


int main()
{
    ArffImporter importer;
    importer.Read( "Dataset/train/train-first10.arff" );

    return 0;
}



