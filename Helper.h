
#ifndef _HELPER_H_
#define _HELPER_H_

#include "BasicDataStructures.h"
#include <stdlib.h>
#include <mpi.h>


using namespace std;
using namespace BasicDataStructures;

namespace MyHelper
{
#define MPI_ERROR_MESSAGE_BUFF_SIZE 50

    int Compare( const void* ele1, const void* ele2 );
    bool StrEqual( const char* str1, const char* str2 );
    // Include string terminator
    unsigned int GetStrLength( const char* str );
    bool IsLetter( const char c );
    Item Tokenize(
        const char* str, 
        const vector<NumericAttr>& featureVec );

    unsigned int getIndexOfMax(
        const unsigned int* uintArray, 
        const unsigned int length );
    // Consume a sorted array, remove duplicates in place, 
    // and return the number of unique elements.
    int removeDuplicates( int* sortedArr, unsigned int length );
    void CheckMPIErr( int errorCode, int mpiNodeId );
}

#endif
