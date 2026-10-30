#include "Python.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits>

#include "svm-module-functions.hpp"
#include "svm.h"
#include "svm-model-python.h"

#define CMD_LEN 2048
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

PyObject* predictLabels(PyObject* features, struct svm_model *model)
{
	PyObject* predicted_labels = Py_BuildValue("[]"), *list_el = NULL;
	PyObject* projections = Py_BuildValue("[]"), *proj_el = NULL;
	double projection;
	int feature_number, testing_instance_number;
	int instance_index;
	double **ptr_instance;
	struct svm_node *x;

	int i, j;

	int svm_type=svm_get_svm_type(model);
	int nr_class=svm_get_nr_class(model);

	feature_number = (int) PyLong_AsLong(PyLong_FromSsize_t(PyList_Size(PyList_GetItem(features, 0)))); /*features colums*/
	testing_instance_number = (int) PyLong_AsLong(PyLong_FromSsize_t(PyList_Size(features))); /*features rows*/

	ptr_instance = Malloc(double*, testing_instance_number);
	for (i = 0; i < testing_instance_number; i++){
		ptr_instance[i] = Malloc(double, feature_number);
		for (j = 0; j < feature_number; j++){
			ptr_instance[i][j] = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(features, i), j));
		}
	}

	x = (struct svm_node*)malloc((feature_number+1)*sizeof(struct svm_node) );
	for(instance_index=0;instance_index<testing_instance_number;instance_index++)
	{
		int i;
		double predict_label;

		for(i=0;i<feature_number;i++)
		{
			x[i].index = i+1;
			x[i].value = ptr_instance[instance_index][i];
		}
		x[feature_number].index = -1;

		if(svm_type == C_RNK ||
                       svm_type == SVORIM ||
		   svm_type == ONE_CLASS ||
		   svm_type == EPSILON_SVR ||
		   svm_type == NU_SVR)
		{
			double res;
			svm_predict_values(model, x, &res);
			/* svm_predict_values already subtracted rho[0] */
			projection = res;
			if(svm_type == ONE_CLASS)
				predict_label =  (res>0)?1:-1;
			else if (svm_type == C_RNK || svm_type == SVORIM){
				int j;
				predict_label = nr_class;
				for(j=1;j<nr_class; j++){
					if (res < model->rho[j]){
						predict_label =  j;
						j=nr_class+1;
					}
				}
			}
			else
				predict_label = res;
		}
		else
		{
			double *dec_values = (double *) malloc(sizeof(double) * nr_class*(nr_class-1)/2);
			svm_predict_values(model, x, dec_values);
			/* one-vs-one voting has no single latent axis */
			projection = std::numeric_limits<double>::quiet_NaN();

			int i;
			int *vote = (int *) malloc(sizeof(int)* nr_class);
			for(i=0;i<nr_class;i++)
				vote[i] = 0;
			int pos=0;
			for(i=0;i<nr_class;i++){
				int j;
				for(j=i+1;j<nr_class;j++)
				{
					if(dec_values[pos++] > 0)
						++vote[i];
					else
						++vote[j];
				}
			}

			int vote_max_idx = 0;
			for(i=1;i<nr_class;i++)
				if(vote[i] > vote[vote_max_idx])
					vote_max_idx = i;
			predict_label = model->label[vote_max_idx];

			free(vote);
			free(dec_values);
		}

		proj_el = Py_BuildValue("d", projection);
		PyList_Append(projections, proj_el);
		Py_DECREF(proj_el);

		list_el = Py_BuildValue("d", predict_label);
		PyList_Append(predicted_labels, list_el);
		//PyList_Append increment the passed in PyObjects references so is necesary
		//to decrement them in order to let python free memory when the model
		//is not longer needed
		Py_DECREF(list_el);
	}

	//Free memory
	free(x);
	for (i = 0; i < testing_instance_number; i++)
		free(ptr_instance[i]);
	free(ptr_instance);

	return Py_BuildValue("(NN)", predicted_labels, projections);
}

/* Interface function of Python*/
PyObject* predict(PyObject* self, PyObject* args){
	struct svm_model *model;
	PyObject* predictedLabels = NULL;

	PyObject* features = NULL;
	PyObject* py_model = NULL;

	/*Parse arguments*/

	/*options is NULL terminated*/
	if (!PyArg_ParseTuple(args, "OO", &features, &py_model)){
		PyErr_SetString(PyExc_RuntimeError, "Unable to parse arguments");
		return NULL;
	}

	if(PyDict_Check(py_model))
	{
		model = pythonToModel(py_model);
		if (model == NULL)
		{
			PyErr_SetString(PyExc_MemoryError, "Unable to translate python model to C");
			return NULL;
		}

		predictedLabels = predictLabels(features, model);
		/* destroy model*/
		svm_free_and_destroy_model(&model);
	}
	else
	{
		PyErr_SetString(PyExc_TypeError, "Model should be a dictionary!");
		return NULL;
	}

	return predictedLabels;
}
