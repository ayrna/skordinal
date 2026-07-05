/**
   boostrank-train.cpp: generates the `fit` function of the python module.
   Based on the original boostrank-train.cpp (c) 2006-2007 Hsuan-Tien Lin
**/
#include "aggrank.h"

#include "orboost.h"
#include "orensemble-model-python.h"
#include "orensemble-module.h"
#include "rankboost.h"
#include "train-functions.h"

PyObject *fit(PyObject *self, PyObject *args)
{
    // Python parameters
    PyObject *labels = NULL;
    PyObject *features = NULL;
    boostrankParams params;

    if (parseArgumentsTrain(args, &features, &labels, params))
    {
        setInvalidArgsErrorTrain();
        return NULL;
    }

    if ((NULL == labels) || (NULL == features))
    {
        PyErr_SetString(PyExc_MemoryError, "Couldn't allocate dataset");
        return NULL;
    }

    /* load training data */
    lemga::DataSet *trd = loadData(features, labels, params);

    if (NULL == trd)
    {
        PyErr_SetString(PyExc_RuntimeError, "Error loading the dataset");
        return NULL;
    }

    lemga::AggRank *pbag = setUpModel(params);

    pbag->set_train_data(trd);
    pbag->train();

    PyObject *ret = modelAndParamsToPython(pbag, params);
    return ret;
}
