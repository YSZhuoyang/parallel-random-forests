
#ifndef _HELPER_H_
#define _HELPER_H_

#include <stdlib.h>
#include <mpi.h>


namespace MyHelper
{
#define MPI_ERROR_MESSAGE_BUFF_SIZE 50

    int Compare( const void* ele1, const void* ele2 );
    bool StrEqual( const char* str1, const char* str2 );
    // Include string terminator
    unsigned int GetStrLength( const char* str );
    unsigned int getIndexOfMax(
        const unsigned int* uintArray, 
        const unsigned int length );
    void RandomizeArray(
        unsigned int* arr, 
        const unsigned int length );
    void CheckMPIErr( int errorCode, int mpiNodeId );
}

#endif
