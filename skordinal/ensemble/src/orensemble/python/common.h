#ifndef __ORENSEMBLE_COMMON_H__
#define __ORENSEMBLE_COMMON_H__
#include "aggrank.h"

typedef struct boostrankParams
{
    UINT bag;
    UINT base;
    UINT n_rank;
    UINT n_iter;
    UINT n_in;
    UINT n_out;
} boostrankParams;

typedef struct boostrankModelParams
{
    lemga::AggRank *model;
    boostrankParams *params;
} boostrankModelParams;
#endif
