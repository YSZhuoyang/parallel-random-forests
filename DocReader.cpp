
#include "DocReader.h"


using namespace std;

DocReader::DocReader()
{
    //state = ReadingState::DOING_NOTHING;
}

DocReader::~DocReader()
{
    for (char* classAttr : classVec) free( classAttr );
    classVec.clear();

    for (char* attr : featureVec) free( attr );
    featureVec.clear();
}

void DocReader::Read( const char* fileName )
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
    char featureName[TOKEN_LENGTH_MAX];
    char featureType[TOKEN_LENGTH_MAX];
    char buffer[READ_LINE_MAX];
    unsigned int value;
    unsigned int numFeatures = 0;
    int readSize;

    while (fgets( buffer, READ_LINE_MAX, fp ) != nullptr)
    {
        // Skip empty line
        if (buffer[0] == '\n')
        {
            continue;
        }

        sscanf( buffer, "%s%n", firstToken, &readSize );

        if (StrEqual( firstToken, KEYWORD_ATTRIBUTE ))
        {
            //sscanf( buffer + readSize, "%s", featureName );
            sscanf( buffer + readSize, "%s %s", featureName, featureType );

            if (StrEqual( featureType, KEYWORD_NUMERIC ))
            {
                printf( "Feature name: %s \n", featureName );

                numFeatures++;
            }
            else
            {
                printf( "Class: %s \n", featureType );

                // Parse classes attributes


            }

            continue;
        }
        else if (StrEqual( firstToken, KEYWORD_DATA))
        {
            unsigned int index;

            while (fgets( buffer, READ_LINE_MAX, fp ) != nullptr)
            {
                index = 0;
                unsigned int featureIndex = 0;
                
                Item* item = new Item;
                item->featureArr = (unsigned int*) malloc( numFeatures * sizeof( unsigned int ) );

                // Get feature attributes
                while (sscanf( buffer + index, "%u%n", &value, &readSize ) > 0)
                {
                    index += readSize + 1;
                    item->featureArr[featureIndex++] = value;

                    //printf( "%u ", value );
                }

                // Get class attributes
                
                //item->classIndex = ;
                
                //printf( "\n" );
            }
            
            break;
        }
    }

    fclose(fp);
}

void DocReader::GetClassAttr( vector<char*>& cv )
{
    cv = classVec;
}

void DocReader::GetAttr( vector<char*>& fv )
{
    fv = featureVec;
}

void DocReader::GetItems( vector<Item*>& iv )
{
    iv = itemVec;
}

bool DocReader::StrEqual( const char str1[], const char str2[] )
{
    unsigned short i = 0;
    while (str1[i] != '\0' && str2[i] != '\0' && str1[i] == str2[i]) i++;

    return (str1[i] == '\0' && str2[i] == '\0') ? true : false;
}
