/**
   boostrank-predict.cpp: main file for performing ordinal regression testing
   with thresholded ensemble models
   (c) 2006-2007 Hsuan-Tien Lin
**/
#include <fstream>
#include <iostream>
#include <set>

#include "aggrank.h"
#include "orboost.h"
#include "rankboost.h"

#include "object.h"

#include "Python.h"
#include "orensemble-model-python.h"

int parseArgumentsPred(PyObject *args, PyObject **features, lemga::AggRank **model, boostrankParams &params);

void setInvalidArgsErrorPred()
{
    PyErr_SetString(PyExc_ValueError, "Usage: model.predict(features)");
}

lemga::DataSet *loadPredData(PyObject *features, const boostrankParams &params)
{
    lemga::DataSet *pd = new lemga::DataSet();
    ssize_t n_test = PyList_Size(features);

    for (ssize_t i = 0; i < n_test; ++i)
    {
        lemga::Input x(params.n_in);
        lemga::Output y(params.n_out);
        if ((UINT)PyList_Size(PyList_GetItem(features, i)) < params.n_in)
        {
            PyErr_SetString(PyExc_ValueError, "Instance has less features than expected");
            delete pd;
            return NULL;
        }
        for (UINT j = 0; j < params.n_in; ++j)
            x[j] = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(features, i), j));

        y[0] = 0;
        pd->append(x, y);
    }

    return pd;
}

PyObject *predict(PyObject *self, PyObject *args)
{

    PyObject *features = NULL;
    boostrankParams params;
    lemga::AggRank *pbag = NULL;
    if (parseArgumentsPred(args, &features, &pbag, params))
    {
        setInvalidArgsErrorPred();
        return NULL;
    }

    if ((NULL == features) || NULL == pbag)
    {
        PyErr_SetString(PyExc_MemoryError, "Couldn't allocate dataset or model");
        return NULL;
    }
    /* load test data */
    lemga::DataSet *td = loadPredData(features, params);
    if (NULL == td)
    {
        delete pbag;
        return NULL;
    }
    lemga::pDataSet ted = td;

    std::vector<lemga::Output> out(ted->size());

    for (UINT i = 0; i < ted->size(); ++i)
        out[i] = (*pbag)(ted->x(i), params.n_iter);

    // std::cout << "Absolute Error: " << ae << std::endl;
    // std::cout << "Classification Error: " << ce << std::endl;
    // std::cout << "Raw Ranking Loss: " << rl << std::endl;
    // std::cout << "Thresholded Ranking Loss: " << tl << std::endl;

    /* out[i][0] is the rank, out[i][1] the ensemble score behind it */
    PyObject *predictedLabels = Py_BuildValue("[]"), *list_el = NULL;
    PyObject *projections = Py_BuildValue("[]"), *proj_el = NULL;
    for (UINT i = 0; i < out.size(); ++i)
    {
        list_el = Py_BuildValue("i", (int)out[i][0]);
        PyList_Append(predictedLabels, list_el);

        Py_DECREF(list_el);

        proj_el = Py_BuildValue("d", (double)out[i][1]);
        PyList_Append(projections, proj_el);

        Py_DECREF(proj_el);
    }

    delete pbag;
    return Py_BuildValue("(NN)", predictedLabels, projections);
}

int parseArgumentsPred(PyObject *args, PyObject **features, lemga::AggRank **model, boostrankParams &params)
{
    PyObject *pyModelParams;
    try
    {
        if (!PyArg_ParseTuple(args, "OO", features, &pyModelParams))
            return 1;

        boostrankModelParams *modelParams = pythonToModelAndParams(pyModelParams);

        if (NULL == modelParams->model || NULL == modelParams->params)
        {
            delete modelParams->model;
            delete modelParams->params;
            delete modelParams;
            return 1;
        }
        params = *modelParams->params;
        *model = modelParams->model;
        delete modelParams->params;
        delete modelParams;
        return 0;
    }

    catch (const std::exception &ex)
    {
        return 1;
    }
}
