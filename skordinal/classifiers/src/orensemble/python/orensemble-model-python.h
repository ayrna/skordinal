#ifndef _ORENSEMBLE_MODEL_PYTHON_HPP_
#define _ORENSEMBLE_MODEL_PYTHON_HPP_

#include "Python.h"
#include "aggrank.h"
#include "common.h"

PyObject *modelToPython(const lemga::AggRank *model);
lemga::AggRank *pythonToModel(PyObject *modelPython);

PyObject *paramsToPython(const boostrankParams &params);
boostrankParams *pythonToParams(PyObject *paramsPyton);

PyObject *modelAndParamsToPython(const lemga::AggRank *model, const boostrankParams &params);
boostrankModelParams *pythonToModelAndParams(PyObject *modelParamsPython);
#endif
