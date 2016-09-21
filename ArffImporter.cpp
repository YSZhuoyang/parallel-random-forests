
#include "ArffImporter.h"


using namespace std;

ArffImporter::ArffImporter()
{
    
}

ArffImporter::~ArffImporter()
{
    for (char* classAttr : classVec) free( classAttr );
    classVec.clear();

    for (NumericAttr& feature : featureVec) free( feature.name );
    featureVec.clear();

    for (Item& item : itemVec) free( item.featureAttrArray );
    itemVec.clear();
}

// Need to check string length boundary
void ArffImporter::Read( const char* fileName )
{
    FILE *fp;

    if ((fp = fopen( fileName, "r+" )) == nullptr)
	{
		printf( "File: %s not found!\n", fileName );
		return;
	}

    // Assuming all data types of all features are integer
    // and ignoring feature types
    char firstToken[TOKEN_LENGTH_MAX];
    char buffer[READ_LINE_MAX];

    while (fgets( buffer, READ_LINE_MAX, fp ) != nullptr)
    {
        // Skip empty lines
        if (buffer[0] == '\n') continue;

        int readSize;
        sscanf( buffer, "%s%n", firstToken, &readSize );

        if (StrEqual( firstToken, KEYWORD_ATTRIBUTE ))
        {
            char* featureName = (char*) malloc( TOKEN_LENGTH_MAX );
            char* featureType = (char*) malloc( TOKEN_LENGTH_MAX );
            
            sscanf( buffer + readSize, "%s %s", featureName, featureType );

            // Read feature names
            if (StrEqual( featureType, KEYWORD_NUMERIC ))
            {
                printf( "Feature name: %s, length: %d \n", featureName, GetStrLength( featureName ) );

                NumericAttr feature;
                feature.name = featureName;
                feature.min = 0;
                feature.max = 0;
                feature.bucketSize = 0;
                // Two buckets by default: 0, greater than 0
                feature.numBuckets = 2;

                featureVec.push_back( feature );
            }
            // Read class names
            else
            {
                // Parse classes attributes
                char* className = (char*) malloc( TOKEN_LENGTH_MAX );
                featureType++;
                
                while (sscanf( featureType, "%[^,}]%n", className, &readSize ) > 0)
                {
                    printf( "Class name: %s \n", className );

                    classVec.push_back( className );
                    className = (char*) malloc( TOKEN_LENGTH_MAX );

                    featureType += readSize + 1;
                }
            }

            continue;
        }
        // Read feature values
        else if (StrEqual( firstToken, KEYWORD_DATA))
        {
            numFeatures = featureVec.size();
            numClasses = classVec.size();
            
            size_t featureAttrArraySize = numFeatures * sizeof( unsigned int );

            while (fgets( buffer, READ_LINE_MAX, fp ) != nullptr)
            {
                unsigned int index = 0;
                unsigned int featureIndex = 0;
                unsigned int value;
                
                Item item;
                item.featureAttrArray = (unsigned int*) malloc( featureAttrArraySize );

                // Get feature attributes
                while (sscanf( buffer + index, "%u%n", &value, &readSize ) > 0)
                {
                    if (featureVec[featureIndex].min > value) featureVec[featureIndex].min = value;
                    if (featureVec[featureIndex].max < value) featureVec[featureIndex].max = value;

                    item.featureAttrArray[featureIndex++] = value;
                    index += readSize + 1;
                }

                // Get class attributes
                char classValue[TOKEN_LENGTH_MAX];
                sscanf( buffer + index, "%s%n", classValue, &readSize );

                for (unsigned short i = 0; i < numClasses; i++)
                {
                    if (StrEqual( classVec[i], classValue ))
                    {
                        item.classIndex = i;
                        break;
                    }
                }

                itemVec.push_back( item );
            }
            
            break;
        }
    }

    fclose(fp);
}

vector<char*> ArffImporter::GetClassAttr()
{
    return classVec;
}

vector<NumericAttr> ArffImporter::GetFeatures()
{
    return featureVec;
}

vector<Item> ArffImporter::GetItems()
{
    return itemVec;
}

bool ArffImporter::StrEqual( const char str1[], const char str2[] )
{
    unsigned short i = 0;
    while (str1[i] != '\0' && str2[i] != '\0' && str1[i] == str2[i]) i++;

    return (str1[i] == '\0' && str2[i] == '\0') ? true : false;
}

unsigned int ArffImporter::GetStrLength( const char* str )
{
    unsigned int len = 0;

    while (str[len++] != '\0');

    return len;
}
