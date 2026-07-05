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
    std::stringstream ss;
    ss << PyBytes_AsString(PyUnicode_AsEncodedString(modelPython, "utf-8", "strict"));
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

PyObject *modelAndParamsToPython(const lemga::AggRank *model, const boostrankParams &params)
{

    PyObject *modelPython = modelToPython(model);

    PyObject *paramsPython = paramsToPython(params);

    PyObject *ret = Py_BuildValue(
        "{s:O, s:O}",
        "model", modelPython,
        "params", paramsPython
    );

    Py_DECREF(modelPython);
    Py_DECREF(paramsPython);

    return ret;
}

boostrankModelParams *pythonToModelAndParams(PyObject *modelParamsPython)
{
    PyObject *pyModel = PyDict_GetItemString(modelParamsPython, "model");
    PyObject *pyParams = PyDict_GetItemString(modelParamsPython, "params");

    boostrankModelParams *ret = new boostrankModelParams();
    ret->model = pythonToModel(pyModel);
    ret->params = pythonToParams(pyParams);

    return ret;
}
