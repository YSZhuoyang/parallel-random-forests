
#include "Helper.h"


int MyHelper::Compare( const void* ele1, const void* ele2 )
{
    int f = *((int*)ele1);
    int s = *((int*)ele2);

    if (f > s) return 1;
    if (f < s) return -1;

    return 0;
}

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

unsigned int MyHelper::getIndexOfMax(
    const unsigned int* uintArray, 
    const unsigned int length )
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

int MyHelper::removeDuplicates(
    int* sortedArr, 
    unsigned int length )
{
    if (sortedArr == nullptr) return 0;

    unsigned int uniqueId = 1;
    unsigned int iter = 1;

    while (iter < length)
    {
        if (sortedArr[iter - 1] != sortedArr[iter])
            sortedArr[uniqueId++] = sortedArr[iter];

        iter++;
    }

    return uniqueId;
}

unsigned int* MyHelper::sampleWithRep(
    unsigned int* container, 
    const unsigned int numSamples, 
    const unsigned int numTotal )
{
    if (numSamples <= 0 || numTotal < numSamples)
    {
        printf( "Number of samples must be greater than 0 and total number\n" );

        return nullptr;
    }

    unsigned int* sampleArr = (unsigned int*)
        malloc( numSamples * sizeof( unsigned int ) );
    unsigned int sampleIndex = 0;
    unsigned int numRest = numTotal - numSamples;

    for (unsigned int i = numTotal - 1; i >= numRest; i--)
    {
        unsigned int randPos = rand() % (i + 1);

        // Swap
        unsigned int temp = container[randPos];
        container[randPos] = container[i];
        container[i] = temp;

        sampleArr[sampleIndex++] = container[i];
    }

    return sampleArr;
}

void MyHelper::randomizeArray(
    unsigned int* arr, 
    const unsigned int length )
{
    for (unsigned int i = length - 1; i > 0; i--)
    {
        unsigned int randPos = rand() % (i + 1);

        // Swap
        unsigned int temp = arr[randPos];
        arr[randPos] = arr[i];
        arr[i] = temp;
    }
}