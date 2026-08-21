#include "orensemble-model-python.h"
#include "common.h"
#include <iostream>
#include <sstream>
#include <string>

PyObject *modelToPython(const lemga::AggRank *model)
{
    std::ostringstream oss;
    oss << *model;
    std::string modelStr = oss.str();
    PyObject *retval = Py_BuildValue("s", modelStr.c_str());
    return retval;
}

PyObject *paramsToPython(const boostrankParams &params)
{
    return Py_BuildValue(
        "kkkkkk",
        params.bag, params.base, params.n_rank,
        params.n_iter, params.n_in, params.n_out
    );
}

lemga::AggRank *pythonToModel(PyObject *modelPython)
{
    PyObject *bytes = PyUnicode_AsEncodedString(modelPython, "utf-8", "strict");
    if (NULL == bytes)
        return NULL;
    std::stringstream ss;
    ss << PyBytes_AsString(bytes);
    Py_DECREF(bytes);
    return (lemga::AggRank *)Object::create(ss);
}

boostrankParams *pythonToParams(PyObject *paramsPyton)
{
    UINT bag, base, n_rank, n_iter, n_in, n_out;
    if (!PyArg_ParseTuple(paramsPyton, "kkkkkk", &bag, &base, &n_rank, &n_iter, &n_in, &n_out))
    {
        PyErr_SetString(PyExc_ValueError, "Couldn't parse parameters");
        return NULL;
    }
    boostrankParams *ret = new boostrankParams();
    ret->bag = bag;
    ret->base = base;
    ret->n_rank = n_rank;
    ret->n_iter = n_iter;
    ret->n_in = n_in;
    ret->n_out = n_out;

    return ret;
}

/* AggRank keeps one set of n_rank-1 thresholds per iteration; take the set
   at the aggregation size prediction will use */
static PyObject *thresholdsToPython(const lemga::AggRank *model, const boostrankParams &params)
{
    UINT iter = params.n_iter;
    if (iter > model->aggregation_size())
        iter = model->aggregation_size();

    PyObject *thresholds = Py_BuildValue("[]"), *el = NULL;
    for (UINT k = 1; k < model->get_n_rank(); ++k)
    {
        el = Py_BuildValue("d", (double)model->threshold(iter, k));
        PyList_Append(thresholds, el);
        Py_DECREF(el);
    }
    return thresholds;
}

PyObject *modelAndParamsToPython(const lemga::AggRank *model, const boostrankParams &params)
{

    PyObject *modelPython = modelToPython(model);

    PyObject *paramsPython = paramsToPython(params);

    PyObject *thresholdsPython = thresholdsToPython(model, params);

    PyObject *ret = Py_BuildValue(
        "{s:O, s:O, s:O}",
        "model", modelPython,
        "params", paramsPython,
        "thresholds", thresholdsPython
    );

    Py_DECREF(modelPython);
    Py_DECREF(paramsPython);
    Py_DECREF(thresholdsPython);

    return ret;
}

boostrankModelParams *pythonToModelAndParams(PyObject *modelParamsPython)
{
    PyObject *pyModel = PyDict_GetItemString(modelParamsPython, "model");
    PyObject *pyParams = PyDict_GetItemString(modelParamsPython, "params");

    boostrankModelParams *ret = new boostrankModelParams();
    ret->model = pyModel ? pythonToModel(pyModel) : NULL;
    ret->params = pyParams ? pythonToParams(pyParams) : NULL;

    return ret;
}
