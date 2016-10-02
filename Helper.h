
#ifndef _HELPER_H_
#define _HELPER_H_

#include <stdlib.h>


namespace MyHelper
{
    int Compare( const void* ele1, const void* ele2 );
    bool StrEqual( const char* str1, const char* str2 );
    // Include string terminator
    unsigned int GetStrLength( const char* str );
    unsigned int getIndexOfMax(
        const unsigned int* uintArray, 
        const unsigned int length );
    void randomizeArray(
        unsigned int* arr, 
        const unsigned int length );
}

#endif
