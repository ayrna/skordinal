#include <stdlib.h>

#include "smo.h"
#include "smo_model_python.h"

PyObject* modelToPython(smo_Settings* model){
	int i, j;
	Data_List* trainlist;
	Data_Node* trainnode;
	PyObject *alpha_list = NULL, *ard_list = NULL, *biasj_list = NULL, *pairs_data_node_list = NULL, *pairs_data_node_dict = NULL, *pairs_point_list = NULL, 
			 *x_mean_list = NULL, *x_devi_list = NULL, *pairs_data_list_dict = NULL, *model_out = NULL, *list_el = NULL;
	
	/*Alpha*/
	alpha_list = Py_BuildValue("[]");

	trainlist = model->pairs;
	trainnode = trainlist->front ;
	j = 0 ;
	while (trainnode!=NULL){
		list_el = Py_BuildValue("d", (model->alpha+j)->alpha);
		PyList_Append(alpha_list, list_el);
		//PyList_Append increment the passed PyObjects references so is necesary
		//to decrement them in order to let python free memory when the model 
		//is not longer needed
		Py_DECREF(list_el);
		trainnode = trainnode->next ;
		j++ ;
	}

	/*Ard*/
	ard_list = Py_BuildValue("[]");

	for(i=0; i<(int)trainlist->dimen; i++){
		list_el = Py_BuildValue("d", model->ard[i]);
		PyList_Append(ard_list, list_el);
		//PyList_Append increment the passed PyObjects references so is necesary
		//to decrement them in order to let python free memory when the model 
		//is not longer needed
		Py_DECREF(list_el);
	}

	/*Biasj*/
	biasj_list = Py_BuildValue("[]");

	for (i=1; i<(int)trainlist->classes; i++){
		list_el = Py_BuildValue("d",model->biasj[i-1]);
		PyList_Append(biasj_list, list_el);
		//PyList_Append increment the passed PyObjects references so is necesary
		//to decrement them in order to let python free memory when the model 
		//is not longer needed
		Py_DECREF(list_el);
	}

	/*Pairs__Data_Nodes*/
	pairs_data_node_list = Py_BuildValue("[]");

	trainnode = trainlist->front ;
	j = 0 ;
	while (trainnode!=NULL){
		/*Point attribute is tranformed to a Python list*/
		pairs_point_list = Py_BuildValue("[]");
		for(i=0; i<(int)trainlist->dimen; i++){
			list_el = Py_BuildValue("d", trainnode->point[i]);
			PyList_Append(pairs_point_list, list_el);
			//PyList_Append increment the passed PyObjects references so is necesary
			//to decrement them in order to let python free memory when the model 
			//is not longer needed
			Py_DECREF(list_el);
		}

		/*The Data_Node structure is transformed to a python dictionary*/
		pairs_data_node_dict = Py_BuildValue(
												"{"
												"s:i,"
												"s:i,"
												"s:i,"
												"s:O,"
												"s:i"
												"}",
												"index", trainnode->index,
												"count", trainnode->count,
												"fold", trainnode->fold,
												"point", pairs_point_list,
												"target", trainnode->target
    	  		 							);
		//Pybuild increment the passed PyObjects references so is necesary
		//to decrement them in order to let python free memory when the model 
		//is not longer needed
		Py_DECREF(pairs_point_list);

		PyList_Append(pairs_data_node_list, pairs_data_node_dict);
		//PyList_Append increment the passed PyObjects references so is necesary
		//to decrement them in order to let python free memory when the model 
		//is not longer needed
		Py_DECREF(pairs_data_node_dict);

		trainnode = trainnode->next ;
		j++ ;
	}

	/*x_mean and x_devi list*/
	x_mean_list = Py_BuildValue("[]");
	x_devi_list = Py_BuildValue("[]");

	for(i=0; i<(int)trainlist->dimen; i++){
		list_el = Py_BuildValue("d", trainlist->x_mean[i]);
		PyList_Append(x_mean_list, list_el);
		//PyList_Append increment the passed PyObjects references so is necesary
		//to decrement them in order to let python free memory when the model 
		//is not longer needed
		Py_DECREF(list_el);

		list_el = Py_BuildValue("d", trainlist->x_devi[i]);
		PyList_Append(x_devi_list, list_el);
		//PyList_Append increment the passed PyObjects references so is necesary
		//to decrement them in order to let python free memory when the model 
		//is not longer needed
		Py_DECREF(list_el);
	}

	/*Pairs_Data_List*/
	pairs_data_list_dict = Py_BuildValue(
											"{"
											"s:i,"
											"s:i,"
											"s:i,"
											"s:i,"
											"s:i,"
											"s:i,"
											"s:i,"
											"s:i,"
											"s:d,"
											"s:d,"
											"s:O,"
											"s:O,"
											"s:O"
											"}",
											"datatype", trainlist->datatype,
											"normalized_input", trainlist->normalized_input,
											"normalized_output", trainlist->normalized_output,
											"count", trainlist->count,
											"dimen", trainlist->dimen,
											"i_ymax", trainlist->i_ymax,
											"i_ymin", trainlist->i_ymin,
											"classes", trainlist->classes,
											"mean", trainlist->mean,
											"deviation", trainlist->deviation,
											"x_mean", x_mean_list,
											"x_devi", x_devi_list,
											"data_nodes_list", pairs_data_node_list
    	  		 						);
	//Pybuild increment the passed PyObjects references so is necesary
	//to decrement them in order to let python free memory when the model 
	//is not longer needed
	Py_DECREF(x_mean_list);
	Py_DECREF(x_devi_list);
	Py_DECREF(pairs_data_node_list);

	/*Create the model as a python dictionary*/
    model_out = Py_BuildValue(
								"{"
								"s:O,"
								"s:i,"
								"s:I,"
								"s:d,"
								"s:O,"
								"s:O,"
								"s:O,"
								"s:d"
								"}",
								"ard", ard_list,
								"kernel", model->kernel,
								"p", model->p,
								"kappa", model->kappa,
								"alpha", alpha_list,
								"pairs", pairs_data_list_dict,
								"biasj", biasj_list,
								"bias", model->bias
    	   					);
	//Pybuild increment the passed PyObjects references so is necesary
	//to decrement them in order to let python free memory when the model 
	//is not longer needed
	Py_DECREF(ard_list);
	Py_DECREF(alpha_list);
	Py_DECREF(pairs_data_list_dict);
	Py_DECREF(biasj_list);

	return model_out;
}

smo_Settings* pythonToModel(PyObject* model){
	PyObject *ard_list = NULL, *pairs_dict = NULL, *x_mean_list = NULL, *x_devi_list = NULL, 
			 *data_nodes_list = NULL, *data_node = NULL, *point_list = NULL, *biasj_list = NULL,
			 *alpha_list = NULL;
	int n, i, j;
	double *ard, *biasj = NULL;
	Alphas * alpha;
	Data_List * pairs = NULL;
	Data_Node *node = NULL;

	smo_Settings* model_out = (smo_Settings*) malloc(sizeof(smo_Settings));
	//Not enough memory
	if(model_out == NULL)
		return NULL;

	//kernel, p, kappa, bias
	model_out->kernel = (int) PyLong_AsLong(PyDict_GetItemString(model, "kernel"));
	model_out->p = (unsigned int) PyLong_AsUnsignedLong(PyDict_GetItemString(model, "p"));
	model_out->kappa = PyFloat_AsDouble(PyDict_GetItemString(model, "kappa"));
	model_out->bias = PyFloat_AsDouble(PyDict_GetItemString(model, "bias"));

	//ard
	ard_list = PyDict_GetItemString(model, "ard");
	n = (int) PyLong_AsLong(PyLong_FromSsize_t(PyList_Size(ard_list)));

	ard = (double*) malloc(n * sizeof(double));
	//Not enough memory
	if(ard == NULL){
		free(model_out);
		return NULL;
	}

	for (i = 0; i < n; i++)
			ard[i] = PyFloat_AsDouble(PyList_GetItem(ard_list, i));

	model_out->ard = ard;

	//pairs
	pairs_dict = PyDict_GetItemString(model, "pairs");
	pairs = (Data_List*) malloc(sizeof(Data_List));
	//Not enough memory
	if(pairs == NULL){
		free(ard);
		free(model_out);
		return NULL;
	}

	//pairs->x_mean
	x_mean_list = PyDict_GetItemString(pairs_dict, "x_mean");
	n = (int) PyLong_AsLong(PyLong_FromSsize_t(PyList_Size(x_mean_list)));
	pairs->x_mean = (double*) malloc(n * sizeof(double));
	//Not enough memory
	if(pairs->x_mean == NULL){
		free(pairs);
		free(ard);
		free(model_out);
		return NULL;
	}
	
	for (i = 0; i < n; i++)
		pairs->x_mean[i] = PyFloat_AsDouble(PyList_GetItem(x_mean_list, i));

	//pairs->x_devi
	x_devi_list = PyDict_GetItemString(pairs_dict, "x_devi");
	n = (int) PyLong_AsLong(PyLong_FromSsize_t(PyList_Size(x_devi_list)));
	pairs->x_devi = (double*) malloc(n * sizeof(double));
	//Not enough memory
	if(pairs->x_devi == NULL){
		free(pairs->x_mean);
		free(pairs);
		free(ard);
		free(model_out);
		return NULL;
	}
	
	for (i = 0; i < n; i++)
			pairs->x_devi[i] = PyFloat_AsDouble(PyList_GetItem(x_devi_list, i));

	//pairs->datatype, normalized_input, normalized_output, count, dimen, i_ymax, i_ymin, classes, mean, deviation
	pairs->datatype = (int) PyLong_AsLong(PyDict_GetItemString(pairs_dict, "datatype"));
	pairs->normalized_input = (int) PyLong_AsLong(PyDict_GetItemString(pairs_dict, "normalized_input"));
	pairs->normalized_output = (int) PyLong_AsLong(PyDict_GetItemString(pairs_dict, "normalized_output"));
	pairs->count = 0;
	pairs->dimen = (int) PyLong_AsLong(PyDict_GetItemString(pairs_dict, "dimen"));
	pairs->i_ymax = (int) PyLong_AsLong(PyDict_GetItemString(pairs_dict, "i_ymax"));
	pairs->i_ymin = (int) PyLong_AsLong(PyDict_GetItemString(pairs_dict, "i_ymin"));
	pairs->classes = (int) PyLong_AsLong(PyDict_GetItemString(pairs_dict, "classes"));
	pairs->mean = PyFloat_AsDouble(PyDict_GetItemString(pairs_dict, "mean"));
	pairs->deviation = PyFloat_AsDouble(PyDict_GetItemString(pairs_dict, "deviation"));

	//pairs->data_nodes_list
	pairs->front = NULL;
	pairs->rear = NULL;

	data_nodes_list = PyDict_GetItemString(pairs_dict, "data_nodes_list");
	n = (int) PyLong_AsLong(PyLong_FromSsize_t(PyList_Size(data_nodes_list)));

	//Get all the list nodes from python
	for (i = 0; i < n; i++){
			data_node = PyList_GetItem(data_nodes_list, i);
			node = (Data_Node *) malloc (sizeof(Data_Node));
			//Not enough memory
			if(node == NULL){
				free(pairs->x_devi);
				free(pairs->x_mean);
				free(pairs);
				free(ard);
				free(model_out);
				return NULL;
			}

			node->index = (int) PyLong_AsLong(PyDict_GetItemString(data_node, "index"));
			node->count = (int) PyLong_AsLong(PyDict_GetItemString(data_node, "count"));
			node->fold = (int) PyLong_AsLong(PyDict_GetItemString(data_node, "fold"));
			node->target = (int) PyLong_AsLong(PyDict_GetItemString(data_node, "target"));
			node->next = NULL;

			node->point = (double*) malloc(pairs->dimen * sizeof(double));
			//Not enough memory
			if(node->point == NULL){
				free(pairs->x_devi);
				free(pairs->x_mean);
				free(pairs);
				free(ard);
				free(model_out);
				return NULL;
			}
			point_list = PyDict_GetItemString(data_node, "point");
			for(j=0; j<(int)pairs->dimen; j++)
				node->point[j] = PyFloat_AsDouble(PyList_GetItem(point_list, j));

			//Add the node to pairs Datalist
			Add_Data_List(pairs, node);
	}

	//pairs->featuretype
	pairs->featuretype = (int *) malloc(pairs->dimen*sizeof(int)) ;
	if (NULL != pairs->featuretype){
		//default 0
		for (i=0; i<(int)pairs->dimen; i++)
			pairs->featuretype[i] = 0 ;
	}
	else{
		free(pairs->x_devi);
		free(pairs->x_mean);
		free(pairs);
		free(ard);
		free(model_out);
		return NULL;
	}

	//Empty values
	pairs->labels = NULL;
	pairs->labelnum = NULL;
	pairs->filename = NULL;

	model_out->pairs = pairs;

	//biasj
	biasj_list = PyDict_GetItemString(model, "biasj");
	n = (int) PyLong_AsLong(PyLong_FromSsize_t(PyList_Size(biasj_list)));

	biasj = (double*) malloc(n * sizeof(double));
	//Not enough memory
	if(biasj == NULL){
		free(pairs->x_devi);
		free(pairs->x_mean);
		free(pairs);
		free(ard);
		free(model_out);
		return NULL;
	}

	for (i = 0; i < n; i++)
		biasj[i] = PyFloat_AsDouble(PyList_GetItem(biasj_list, i));

	model_out->biasj = biasj;

	//alpha
	alpha_list = PyDict_GetItemString(model, "alpha");
	n = (int) PyLong_AsLong(PyLong_FromSsize_t(PyList_Size(alpha_list)));
	alpha = (Alphas*) malloc(n * sizeof(Alphas));
	//Not enough memory
	if(alpha == NULL){
		free(biasj);
		free(pairs->x_devi);
		free(pairs->x_mean);
		free(pairs);
		free(ard);
		free(model_out);
		return NULL;
	}

	for (i = 0; i < n; i++)
		(alpha + i)->alpha = PyFloat_AsDouble(PyList_GetItem(alpha_list, i));

	model_out->alpha = alpha;

	return model_out;
}
