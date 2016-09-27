
#include "Helper.h"


bool MyHelper::StrEqual( const char* str1, const char* str2 )
{
    unsigned short i = 0;
    while (str1[i] != '\0' && str2[i] != '\0' && str1[i] == str2[i]) i++;

    return (str1[i] == '\0' && str2[i] == '\0') ? true : false;
}

unsigned int MyHelper::GetStrLength( const char* str )
{
    unsigned int len = 0;

    while (str[len++] != '\0');

    return len;
}

unsigned int MyHelper::getIndexOfMax( const unsigned int* uintArray, const unsigned int length )
{
    if (uintArray == nullptr || length <= 0) return 0;

    unsigned int indexOfMax = 0;
    unsigned int max = 0;

    for (unsigned int i = 0; i < length; i++)
    {
        if (uintArray[i] > max)
        {
            indexOfMax = i;
            max = uintArray[i];
        }
    }

    return indexOfMax;
}

void MyHelper::randomizeArray( unsigned int* arr, const unsigned int length, const unsigned numSwap )
{
    for (unsigned int i = 0; i < numSwap; i++)
    {
        unsigned int randomA = rand() % length;
        unsigned int randomB = rand() % length;

        // Swap
        unsigned int temp = arr[randomA];
        arr[randomA] = arr[randomB];
        arr[randomB] = temp;
    }
}
