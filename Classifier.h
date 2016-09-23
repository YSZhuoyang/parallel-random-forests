
#ifndef _CLASSIFIER_H_
#define _CLASSIFIER_H_

#include "TreeBuilder.h"

class Classifier
{
public:
    Classifier();
    ~Classifier();

    void Train();
    void Classify();

private:
    TreeBuilder treeBuilder;
};

#endif
