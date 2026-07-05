#ifndef _ORENSEMBLE_TRAIN_COMMON
#define _ORENSEMBLE_TRAIN_COMMON

#include "Python.h"
#include "common.h"

void setInvalidArgsErrorTrain();
int parseArgumentsTrain(PyObject *args, PyObject **features, PyObject **labels, boostrankParams &params);
lemga::LearnModel *setUpBaseLearner(const boostrankParams &params);
lemga::AggRank *setUpModel(const boostrankParams &params);
lemga::DataSet *loadData(PyObject *features, PyObject *labels, const boostrankParams &params);

#endif
