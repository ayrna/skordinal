#include "train-functions.h"
#include "softperc.h"

#include <feedforwardnn.h>
#include <nnlayer.h>
#include <perceptron.h>
#include <stump.h>

#include "aggrank.h"
#include "orboost.h"
#include "rankboost.h"

void setInvalidArgsErrorTrain()
{
    PyErr_SetString(
        PyExc_ValueError,
        "Usage: model = orensemble.fit( training_labels training_instances bag base n_rank n_iter\n"
        "bag : 10 = RankBoost, cla_thres, reg_param=0.0\n"
        "      11 = RankBoost, cla_thres, reg_param=1e-32\n"
        "      20 = RankBoost, abs_thres, reg_param=0.0\n"
        "      21 = RankBoost, abs_thres, reg_param=1e-32\n"
        "      30 = ORBoost, FORM_LR, ordered, sub_iter=1, reg_param=0.0\n"
        "      31 = ORBoost, FORM_LR, ordered, sub_iter=1, reg_param=1e-32\n"
        "      40 = ORBoost, FORM_FULL, ordered, sub_iter=1, reg_param=0.0\n"
        "      41 = ORBoost, FORM_FULL, ordered, sub_iter=1, reg_param=1e-32\n"
        "base: 100 = stump (without constant)\n"
        "      200 = perc200 with special bias\n"
        "      201 = perc200 with special bias, scale=1\n"
        "      204 = perc200 with special bias, scale=4\n"
        "      3HH = neural net, number of hidden neurons=HH\n"
    );
}

int parseArgumentsTrain(PyObject *args, PyObject **features, PyObject **labels, boostrankParams &params)
{
    UINT bag, base, n_rank, n_iter;
    try
    {
        if (!PyArg_ParseTuple(args, "OOkkkk", features, labels, &bag, &base, &n_rank, &n_iter))
            return 1;

        params.bag = bag;
        params.base = base;
        params.n_rank = n_rank;
        params.n_iter = n_iter;

        params.n_in = PyLong_AsUnsignedLong(PyLong_FromSsize_t(PyList_Size(PyList_GetItem(*features, 0))));
        params.n_out = 1;
        return 0;
    }

    catch (const std::exception &ex)
    {
        return 1;
    }
}
lemga::LearnModel *setUpBaseLearner(const boostrankParams &params)
{
    lemga::LearnModel *pst = 0;

    switch (params.base)
    {
    case 100:
    {
        lemga::Stump *p = new lemga::Stump(params.n_in);
        pst = p;
        break;
    }
    case 200:
    {
        lemga::Perceptron *p = new lemga::Perceptron(params.n_in);
        p->set_parameter(0, 0, 200);
        p->set_train_method(lemga::Perceptron::RAND_COOR_DESCENT_BIAS);
        pst = p;
        break;
    }
    case 201:
    {
        lemga::SoftPerc *p = new lemga::SoftPerc(params.n_in);
        p->set_parameter(0, 0, 200);
        p->set_train_method(lemga::Perceptron::RAND_COOR_DESCENT_BIAS);
        p->set_scale(1);
        pst = p;
        break;
    }
    case 204:
    {
        lemga::SoftPerc *p = new lemga::SoftPerc(params.n_in);
        p->set_parameter(0, 0, 200);
        p->set_train_method(lemga::Perceptron::RAND_COOR_DESCENT_BIAS);
        p->set_scale(4);
        pst = p;
        break;
    }
    default:
        if (params.base / 300 != 1)
        {
            PyErr_SetString(PyExc_ValueError, "Invalid base learner\n");
            return NULL;
        }
        lemga::FeedForwardNN *p = new lemga::FeedForwardNN();
        lemga::NNLayer l(params.n_in, params.base % 300);
        l.set_weight_range(-1, 1);
        p->add_top(l);
        lemga::NNLayer lout(params.base % 300, 1);
        lout.set_weight_range(-1, 1);
        p->add_top(lout);
        p->set_train_method(lemga::FeedForwardNN::CONJUGATE_GRADIENT);
        p->set_parameter(0.1, 1e-7, 200); // 1e-10, 2000
        p->initialize();
        pst = p;
    }
    return pst;
}

lemga::DataSet *loadData(PyObject *features, PyObject *labels, const boostrankParams &params)
{
    lemga::DataSet *pd = new lemga::DataSet();

    ssize_t instance_number = PyLong_AsLong(PyLong_FromSsize_t(PyList_Size(features))); /*features rows*/
    ssize_t label_number = PyLong_AsLong(PyLong_FromSsize_t(PyList_Size(labels)));

    if (instance_number != label_number)
    {
        PyErr_SetString(PyExc_ValueError, "Number of labels is different to the number of instances");
        return NULL;
    }

    for (ssize_t i = 0; i < instance_number; ++i)
    {
        lemga::Input x(params.n_in);
        lemga::Output y(params.n_out);

        if (PyLong_AsUnsignedLong(PyLong_FromSsize_t(PyList_Size(PyList_GetItem(features, i)))) < params.n_in)
        {
            PyErr_SetString(PyExc_ValueError, "Instance has less features than expected");
            return NULL;
        }
        for (UINT j = 0; j < params.n_in; ++j)
            x[j] = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(features, i), j));

        REAL out = PyFloat_AsDouble(PyList_GetItem(labels, i));
        if (out < 1)
        {
            PyErr_SetString(PyExc_ValueError, "Found invalid (0) label");
            return NULL;
        }
        y[0] = out;
        pd->append(x, y);
    }

    return pd;
}

lemga::AggRank *setUpModel(const boostrankParams &params)
{
    lemga::LearnModel *pst = setUpBaseLearner(params);

    if (NULL == pst)
        return NULL;
    lemga::AggRank *pbag = 0;

    switch (params.bag / 10)
    {
    case 1:
    {
        lemga::RankBoost *p = new lemga::RankBoost(params.n_rank);
        p->set_thres_mode(lemga::AggRank::THRES_CLALOSS);
        pbag = p;
        break;
    }
    case 2:
    {
        lemga::RankBoost *p = new lemga::RankBoost(params.n_rank);
        p->set_thres_mode(lemga::AggRank::THRES_ABSLOSS);
        pbag = p;
        break;
    }
    case 3:
    {
        lemga::ORBoost *p = new lemga::ORBoost(params.n_rank);
        p->set_sub_iter(1);
        p->set_ordered(true);
        p->set_form(lemga::ORBoost::FORM_LR);
        pbag = p;
        break;
    }
    case 4:
    {
        lemga::ORBoost *p = new lemga::ORBoost(params.n_rank);
        p->set_sub_iter(1);
        p->set_ordered(true);
        p->set_form(lemga::ORBoost::FORM_FULL);
        pbag = p;
        break;
    }
    }

    if (params.bag % 10 == 1)
        pbag->set_reg_param(1e-32);
    else
        pbag->set_reg_param(0.0);

    pbag->set_base_model(*pst);
    pbag->set_max_models(params.n_iter);
    pbag->reset();

    return pbag;
}
