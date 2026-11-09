#include "Python.h"

#include <stdlib.h>
#include <ctype.h>

#include "svm-module-functions.hpp"
#include "svm.h"
#include "svm-model-python.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

void print_null(const char *s) {}

const char* read_problem(PyObject* label_vec, PyObject* instance_mat);

/* svm arguments*/
struct svm_parameter param;		/* set by fit*/
struct svm_problem prob;		/* set by read_problem*/
struct svm_model *model;
struct svm_node *x_space;

/* Interface function of Python*/
PyObject* fit(PyObject* self, PyObject* args, PyObject* kwargs){
	const char *error_msg;
	static const char *kwlist[] = {"", "", "kernel", "degree", "gamma", "coef0",
		"tol", "C", "shrinking", "cache_size", NULL};

	/* fix random seed to have same results for each run*/
	/* (for cross validation and probability estimation)*/
	srand(1);

	/* Transform the input Matrix to libsvm format*/
	const char *err;

	PyObject* labels = NULL;
	PyObject* features = NULL;

	// Take the hyperparameters as C values, as scikit-learn's libsvm.fit does
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOiiddddpd",
			const_cast<char**>(kwlist), &labels, &features, &param.kernel_type,
			&param.degree, &param.gamma, &param.coef0, &param.eps, &param.C,
			&param.shrinking, &param.cache_size))
		return NULL;
	// Fix C_RNK and libsvm's defaults for the options REDSVM does not expose
	param.svm_type = C_RNK;
	param.nu = 0.5;
	param.p = 0.1;
	param.probability = 0;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	svm_set_print_string_function(&print_null);

	err = read_problem(labels, features);

	/* svmtrain's original code*/
	error_msg = svm_check_parameter(&prob, &param);

	if(err || error_msg)
	{
		svm_destroy_param(&param);
		free(prob.y);
		free(prob.x);
		free(x_space);
		if (error_msg != NULL)
			PyErr_SetString(PyExc_ValueError, error_msg);
		else
			PyErr_SetString(PyExc_ValueError, err);
		return NULL;
	}
	PyObject*  py_model;

	model = svm_train(&prob, &param);

	if (model == NULL){
		PyErr_SetString(PyExc_ValueError, "The trained model is null");
		svm_destroy_param(&param);
		free(prob.y);
		free(prob.x);
		free(x_space);
		return NULL;
	}

	py_model = modelToPython(model);

	svm_free_and_destroy_model(&model);

	svm_destroy_param(&param);
	free(prob.y);
	free(prob.x);
	free(x_space);

	return py_model;
}

const char* read_problem(PyObject* label_vec, PyObject* instance_mat)
{
	int i, j, k;
	int elements, max_index, sc, label_vector_row_num;
	double **samples, *labels;

	prob.x = NULL;
	prob.y = NULL;
	x_space = NULL;

	sc = (int) PyList_Size(PyList_GetItem(instance_mat, 0)); /*instance_mat colums*/

	elements = 0;
	/* the number of instance*/
	prob.l = (int) PyList_Size(instance_mat); /*instance_mat rows*/
	label_vector_row_num = (int) PyList_Size(label_vec);

	if(label_vector_row_num!=prob.l)
	{
		return "Length of label vector does not match # of instances";
	}

	//PyObjects are transformed to C arrays
	labels = Malloc(double, label_vector_row_num);
	for (i = 0; i < label_vector_row_num; i++){
		labels[i] = PyFloat_AsDouble(PyList_GetItem(label_vec, i));
	}

	samples = Malloc(double*, prob.l);
	for (i = 0; i < prob.l; i++){
		samples[i] = Malloc(double, sc);
		for (j = 0; j < sc; j++){
			samples[i][j] = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(instance_mat, i), j));
		}
	}

	if(param.kernel_type == PRECOMPUTED)
		elements = prob.l * (sc + 1);
	else
	{
		for(i = 0; i < prob.l; i++)
		{
			for(k = 0; k < sc; k++)
				if(samples[i][k] != 0)
					elements++;
			/* count the '-1' element*/
			elements++;
		}
	}

	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct svm_node *,prob.l);
	x_space = Malloc(struct svm_node, elements);

	max_index = sc;
	j = 0;
	for(i = 0; i < prob.l; i++)
	{
		prob.x[i] = &x_space[j];
		prob.y[i] = labels[i];

		for(k = 0; k < sc; k++)
		{
			if(param.kernel_type == PRECOMPUTED || samples[i][k] != 0)
			{
				x_space[j].index = k + 1;
				x_space[j].value = samples[i][k];
				j++;
			}
		}
		x_space[j++].index = -1;
	}

	//Free memory
	free(labels);
	for(i = 0; i < prob.l; i++)
		free(samples[i]);
	free(samples);

	if(param.gamma == 0 && max_index > 0)
		param.gamma = 1.0/max_index;

	if(param.kernel_type == PRECOMPUTED)
		for(i=0;i<prob.l;i++)
		{
			if((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
			{
				return "Wrong input format: sample_serial_number out of range";
			}
		}

	return NULL;
}
