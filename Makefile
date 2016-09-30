

################################ Macros #################################

SHELL = /bin/sh
CFLAGS = -g -std=c++11 -Wall
CC = mpic++
OBJECTS = Helper.o ArffImporter.o TreeBuilder.o Classifier.o

################################ Compile ################################

exec: ${OBJECTS} Main.c
	$(CC) ${CFLAGS} -o $@ ${OBJECTS} Main.c

Helper.o: Helper.c Helper.h
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
