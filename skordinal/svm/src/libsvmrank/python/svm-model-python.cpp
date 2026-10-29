#include <stdlib.h>

#include "svm.h"
#include "svm-model-python.h"

PyObject* modelToPython(struct svm_model* model){
	int n, nr_class, i, j, x_index;
	int *label = NULL, *nSV = NULL;
	double *rho = NULL, *probA = NULL, *probB = NULL;
	double** sv_coef = NULL;
	PyObject *param_dict = NULL, *rho_list = NULL, *model_out = NULL, *label_list = NULL, *probA_list = NULL, *probB_list = NULL,
						   *nSV_list = NULL, *sv_coef_list = NULL, *sv_coef_listAux = NULL, *svmNode_dict = NULL, *svmNode_list = NULL,
						   *svmNode_listAux = NULL, *list_el = NULL;
	struct svm_parameter* param = NULL;
	struct svm_node** SV = NULL;

	/*param*/
	param = &(model->param);

	param_dict = Py_BuildValue(
    			"{"
    			"s:i,"
    			"s:i,"
    			"s:i,"
    			"s:d,"
    			"s:d"
    			"}",
    			"svm_type", param->svm_type,
    			"kernel_type", param->kernel_type,
    			"degree", param->degree,
    			"gamma", param->gamma,
    			"coef0", param->coef0
    	   );


	/* rho*/
	rho = model->rho;
	rho_list = Py_BuildValue("[]");

	if ((model->param).svm_type != C_RNK && (model->param).svm_type != SVORIM){
		n = model->nr_class * (model->nr_class - 1) / 2;
		for(i = 0; i < n; i++){
			list_el = Py_BuildValue("d", rho[i]);
			PyList_Append(rho_list, list_el);
			//PyList_Append increment the passed PyObjects references so is necesary
			//to decrement them in order to let python free memory when the model
			//is not longer needed
			Py_DECREF(list_el);
		}

		nr_class = model->nr_class;
	}
	else{
		n = model->nr_class;
		for(i = 0; i < n; i++){
			list_el = Py_BuildValue("d", rho[i]);
			PyList_Append(rho_list, list_el);
			//PyList_Append increment the passed PyObjects references so is necesary
			//to decrement them in order to let python free memory when the model
			//is not longer needed
			Py_DECREF(list_el);
		}
		nr_class = 2;
	}


	/* Label*/
	label = model->label;
	label_list = Py_BuildValue("[]");

	if(model->label != NULL){
		for (i = 0; i < nr_class; i++){
			list_el = Py_BuildValue("i", label[i]);
			PyList_Append(label_list, list_el);
			//PyList_Append increment the passed PyObjects references so is necesary
			//to decrement them in order to let python free memory when the model
			//is not longer needed
			Py_DECREF(list_el);
		}
	}


	/* probA*/
	probA = model->probA;
	probA_list = Py_BuildValue("[]");

	if(model->probA != NULL){
		for (i = 0; i < n; i++){
			list_el = Py_BuildValue("d", probA[i]);
			PyList_Append(probA_list, list_el);
			//PyList_Append increment the passed PyObjects references so is necesary
			//to decrement them in order to let python free memory when the model
			//is not longer needed
			Py_DECREF(list_el);
		}
	}


	/* probB*/
	probB = model->probB;
	probB_list = Py_BuildValue("[]");

	if(model->probA != NULL){
		for (i = 0; i < n; i++){
			list_el = Py_BuildValue("d", probB[i]);
			PyList_Append(probB_list, list_el);
			//PyList_Append increment the passed PyObjects references so is necesary
			//to decrement them in order to let python free memory when the model
			//is not longer needed
			Py_DECREF(list_el);
		}
	}


	/* nSV*/
	nSV = model->nSV;
	nSV_list = Py_BuildValue("[]");

	if(model->nSV != NULL){
		for (i = 0; i < nr_class; i++){
			list_el = Py_BuildValue("i", nSV[i]);
			PyList_Append(nSV_list, list_el);
			//PyList_Append increment the passed PyObjects references so is necesary
			//to decrement them in order to let python free memory when the model
			//is not longer needed
			Py_DECREF(list_el);
		}
	}


	/* sv_coef*/
	sv_coef = (model->sv_coef);
	sv_coef_list = Py_BuildValue("[]");

	for (i = 0; i < nr_class-1; i++){
		sv_coef_listAux = Py_BuildValue("[]");
		for (j = 0; j < model->l; j++){
			list_el = Py_BuildValue("d", sv_coef[i][j]);
			PyList_Append(sv_coef_listAux, list_el);
			//PyList_Append increment the passed PyObjects references so is necesary
			//to decrement them in order to let python free memory when the model
			//is not longer needed
			Py_DECREF(list_el);
		}

		PyList_Append(sv_coef_list, sv_coef_listAux);
		//PyList_Append increment the passed PyObjects references so is necesary
		//to decrement them in order to let python free memory when the model
		//is not longer needed
		Py_DECREF(sv_coef_listAux);
	}


	/*SV*/
	svmNode_list = Py_BuildValue("[]");
	SV = model->SV;

	for(i = 0;i < model->l; i++){
		svmNode_listAux = Py_BuildValue("[]");

		if(model->param.kernel_type == PRECOMPUTED){
			/* make a (model->l x 1) matrix*/
			svmNode_dict = Py_BuildValue(
					"{"
					"s:i,"
					"s:d"
					"}",
					"index", 0,
					"value", SV[i][0].value
			);

			PyList_Append(svmNode_listAux, svmNode_dict);
			//PyList_Append increment the passed PyObjects references so is necesary
			//to decrement them in order to let python free memory when the model
			//is not longer needed
			Py_DECREF(svmNode_dict);
		}
		else{
			x_index = 0;
			while (model->SV[i][x_index].index != -1){
				svmNode_dict = Py_BuildValue(
					"{"
					"s:i,"
					"s:d"
					"}",
					"index", SV[i][x_index].index,
					"value", SV[i][x_index].value
				);

				PyList_Append(svmNode_listAux, svmNode_dict);
				//PyList_Append increment the passed PyObjects references so is necesary
				//to decrement them in order to let python free memory when the model
				//is not longer needed
				Py_DECREF(svmNode_dict);
				x_index++;
			}
		}

		PyList_Append(svmNode_list, svmNode_listAux);
		//PyList_Append increment the passed PyObjects references so is necesary
		//to decrement them in order to let python free memory when the model
		//is not longer needed
		Py_DECREF(svmNode_listAux);
	}

	/*Create the model as a python dictionary*/
    model_out = Py_BuildValue(
    			"{"
    			"s:O,"
    			"s:i,"
    			"s:i,"
    			"s:O,"
    			"s:O,"
    			"s:O,"
    			"s:O,"
    			"s:O,"
    			"s:O,"
    			"s:O"
    			"}",
    			"param", param_dict,
    			"nr_class", model->nr_class,
    			"l", model->l,
    			"SV", svmNode_list,
    			"sv_coef", sv_coef_list,
    			"rho", rho_list,
    			"probA", probA_list,
    			"probB", probB_list,
    			"label", label_list,
    			"nSV", nSV_list
    	   );
	//Pybuild increment the passed PyObjects references so is necesary
	//to decrement them in order to let python free memory when the model
	//is not longer needed
	Py_DECREF(param_dict);
	Py_DECREF(svmNode_list);
	Py_DECREF(sv_coef_list);
	Py_DECREF(rho_list);
	Py_DECREF(probA_list);
	Py_DECREF(probB_list);
	Py_DECREF(label_list);
	Py_DECREF(nSV_list);

	return model_out;
}

struct svm_model* pythonToModel(PyObject* model){
	struct svm_model* model_out = (struct svm_model*) malloc(sizeof(struct svm_model));
	//Not enough memory
	if(model_out == NULL){
		return NULL;
	}

	PyObject *param = NULL, *SV_list = NULL, *svCoef_list = NULL, *aux_list = NULL, *rho_list = NULL, *probA_list = NULL,
			 *probB_list = NULL, *label_list = NULL, *nSV_list = NULL;
	int n, nAux, i, j;

	/*param*/
	param =  PyDict_GetItemString(model, "param");

	(model_out->param).svm_type = (int) PyLong_AsLong(PyDict_GetItemString(param, "svm_type"));
	(model_out->param).kernel_type = (int) PyLong_AsLong(PyDict_GetItemString(param, "kernel_type"));
	(model_out->param).degree = (int) PyLong_AsLong(PyDict_GetItemString(param, "degree"));
	(model_out->param).gamma = PyFloat_AsDouble(PyDict_GetItemString(param, "gamma"));
	(model_out->param).coef0 = PyFloat_AsDouble(PyDict_GetItemString(param, "coef0"));

	/*nr_class*/
	model_out->nr_class = (int) PyLong_AsLong(PyDict_GetItemString(model, "nr_class"));

	/*l*/
	model_out->l = (int) PyLong_AsLong(PyDict_GetItemString(model, "l"));

	/*SV*/
	SV_list = PyDict_GetItemString(model, "SV");
	aux_list = NULL;
	n = (int) PyLong_AsLong(PyLong_FromSsize_t(PyList_Size(SV_list)));

	model_out->SV = (struct svm_node**) malloc(n * sizeof(struct svm_node*));
	//Not enough memory
	if(model_out->SV == NULL){
		free(model_out);
		return NULL;
	}

	for (i = 0; i < n; i++){
		aux_list = PyList_GetItem(SV_list, i);

		nAux = (int) PyLong_AsLong(PyLong_FromSsize_t(PyList_Size(aux_list)));
		(model_out->SV)[i] = (struct svm_node*) malloc((nAux+1) * sizeof(struct svm_node));
		//Not enough memory
		if((model_out->SV)[i] == NULL){
			free(model_out->SV);
			free(model_out);
			return NULL;
		}

		for (j = 0; j < nAux; j++){
			((model_out->SV)[i][j]).index = (int) PyLong_AsLong(PyDict_GetItemString(PyList_GetItem(aux_list, j), "index"));
			((model_out->SV)[i][j]).value = PyFloat_AsDouble(PyDict_GetItemString(PyList_GetItem(aux_list, j), "value"));
		}

		//Add the finish element
		((model_out->SV)[i][nAux]).index = -1;
	}

	/*sv_coef*/
	svCoef_list = PyDict_GetItemString(model, "sv_coef");
	aux_list = NULL;
	n = (int) PyLong_AsLong(PyLong_FromSsize_t(PyList_Size(svCoef_list)));

	model_out->sv_coef = (double**) malloc(n * sizeof(double*));
	//Not enough memory
	if(model_out->sv_coef == NULL){
		free(model_out->SV);
		free(model_out);
		return NULL;
	}

	for (i = 0; i < n; i++){
		aux_list = PyList_GetItem(svCoef_list, i);

		nAux = (int) PyLong_AsLong(PyLong_FromSsize_t(PyList_Size(aux_list)));
		(model_out->sv_coef)[i] = (double*) malloc(nAux * sizeof(double));
		//Not enough memory
		if((model_out->sv_coef)[i] == NULL){
			free(model_out->sv_coef);
			free(model_out->SV);
			free(model_out);
			return NULL;
		}

		for (j = 0; j < nAux; j++){
			(model_out->sv_coef)[i][j] = PyFloat_AsDouble(PyList_GetItem(aux_list, j));
		}
	}

	/*rho*/
	rho_list = PyDict_GetItemString(model, "rho");
	n = (int) PyLong_AsLong(PyLong_FromSsize_t(PyList_Size(rho_list)));

	model_out->rho = (double*) malloc(n * sizeof(double));
	//Not enough memory
	if(model_out->rho == NULL){
		free(model_out->sv_coef);
		free(model_out->SV);
		free(model_out);
		return NULL;
	}

	for (i = 0; i < n; i++){
		(model_out->rho)[i] = PyFloat_AsDouble(PyList_GetItem(rho_list, i));
	}

	/*probA*/
	probA_list = PyDict_GetItemString(model, "probA");
	n = (int) PyLong_AsLong(PyLong_FromSsize_t(PyList_Size(probA_list)));

	model_out->probA = (double*) malloc(n * sizeof(double));
	//Not enough memory
	if(model_out->probA == NULL){
		free(model_out->rho);
		free(model_out->sv_coef);
		free(model_out->SV);
		free(model_out);
		return NULL;
	}

	for (i = 0; i < n; i++){
		(model_out->probA)[i] = PyFloat_AsDouble(PyList_GetItem(probA_list, i));
	}

	/*probB*/
	probB_list = PyDict_GetItemString(model, "probB");
	n = (int) PyLong_AsLong(PyLong_FromSsize_t(PyList_Size(probB_list)));

	model_out->probB = (double*) malloc(n * sizeof(double));
	//Not enough memory
	if(model_out->probB == NULL){
		free(model_out->probA);
		free(model_out->rho);
		free(model_out->sv_coef);
		free(model_out->SV);
		free(model_out);
		return NULL;
	}

	for (i = 0; i < n; i++){
		(model_out->probB)[i] = PyFloat_AsDouble(PyList_GetItem(probB_list, i));
	}

	/*label*/
	label_list = PyDict_GetItemString(model, "label");
	n = (int) PyLong_AsLong(PyLong_FromSsize_t(PyList_Size(label_list)));

	model_out->label = (int*) malloc(n * sizeof(int));
	//Not enough memory
	if(model_out->label == NULL){
		free(model_out->probB);
		free(model_out->probA);
		free(model_out->rho);
		free(model_out->sv_coef);
		free(model_out->SV);
		free(model_out);
		return NULL;
	}

	for (i = 0; i < n; i++){
		(model_out->label)[i] = (int) PyLong_AsLong(PyList_GetItem(label_list, i));;
	}

	/*nSV*/
	nSV_list = PyDict_GetItemString(model, "nSV");
	n = (int) PyLong_AsLong(PyLong_FromSsize_t(PyList_Size(nSV_list)));

	model_out->nSV = (int*) malloc(n * sizeof(int));
	//Not enough memory
	if(model_out->nSV == NULL){
		free(model_out->label);
		free(model_out->probB);
		free(model_out->probA);
		free(model_out->rho);
		free(model_out->sv_coef);
		free(model_out->SV);
		free(model_out);
		return NULL;
	}

	for (i = 0; i < n; i++){
		(model_out->nSV)[i] = (int) PyLong_AsLong(PyList_GetItem(nSV_list, i));;
	}

	/*free_sv*/
	model_out->free_sv = 1;

	return model_out;
}
