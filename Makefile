

################################ Macros #################################

SHELL = /bin/sh
# Enable debug options
# CFLAGS = -g -Wall
# Enable best optimization options
CFLAGS = -Ofast -march=native -mtune=native -std=c++11
CC = g++
OBJECTS = Helper.o ArffImporter.o TreeBuilder.o Classifier.o

################################ Compile ################################

exec: ${OBJECTS} Main.c
	$(CC) ${CFLAGS} -o $@ ${OBJECTS} Main.c

Helper.o: Helper.c Helper.h BasicDataStructures.h
	$(CC) ${CFLAGS} -c Helper.c

ArffImporter.o: ArffImporter.cpp ArffImporter.h BasicDataStructures.h Helper.h
	$(CC) ${CFLAGS} -c ArffImporter.cpp

TreeBuilder.o: TreeBuilder.cpp TreeBuilder.h BasicDataStructures.h Helper.h
	$(CC) ${CFLAGS} -c TreeBuilder.cpp

Classifier.o: Classifier.cpp Classifier.h TreeBuilder.h
	$(CC) ${CFLAGS} -c Classifier.cpp

################################# Clean #################################

clean:
	-rm -f *.o *.h.gch exec
