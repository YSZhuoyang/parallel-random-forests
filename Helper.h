
#ifndef _HELPER_H_
#define _HELPER_H_

#include <stdio.h>
#include <stdlib.h>
#include <vector>


using namespace std;

namespace MyHelper
{
    int Compare( const void* ele1, const void* ele2 );
    bool StrEqual( const char* str1, const char* str2 );
    // Include string terminator
    unsigned int GetStrLength( const char* str );
    unsigned int getIndexOfMax(
        const unsigned int* uintArray, 
        const unsigned int length );
    unsigned int* sampleWithRep(
        unsigned int* container, 
        const unsigned int numSamples, 
        unsigned int& numRest );
    void randomizeArray(
        unsigned int* arr, 
        const unsigned int length );
}

#endif
