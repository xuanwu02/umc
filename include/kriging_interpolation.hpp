#ifndef _UMC_KRIGING_HPP
#define _UMC_KRIGING_HPP

#include <iostream>
#include <vector>
#include <math.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_multifit_nlin.h>
#include "adjacent_prediction.hpp"

namespace UMC{

struct sampleData{
    size_t m;
    std::vector<double> lag;
    std::vector<double> semivariogram;
};

int model_f(const gsl_vector * x, void * params, gsl_vector * f){
    double range = gsl_vector_get(x, 0);
    double scale = gsl_vector_get(x, 1);
    sampleData *sdata = (sampleData *)params;

    for(size_t i=0; i<sdata->m; i++){
        double lag = sdata->lag[i];
        double semivar = sdata->semivariogram[i];
        double residual = semivar - (scale * (1.0 - exp(- (lag*lag) / (range*range))));
        // double residual = semivar - (scale * (1.0 - exp(- lag / range)));
        gsl_vector_set(f, i, residual);
    }

    return GSL_SUCCESS;
}

void fit_model(const std::vector<double>& lag, const std::vector<double>& semivariogram, double& range_est, double& scale_est, const int max_iter){
    size_t m = lag.size();

    // Initialize GSL solver
    const gsl_multifit_fdfsolver_type *T;
    gsl_multifit_fdfsolver *s;
    int status;
    size_t iter = 0;
    const size_t p = 2; 

    T = gsl_multifit_fdfsolver_lmsder;
    s = gsl_multifit_fdfsolver_alloc(T, m, p);
    gsl_vector *x = gsl_vector_alloc(p);
    gsl_vector_set(x, 0, range_est);
    gsl_vector_set(x, 1, scale_est);

    sampleData sdata;
    sdata.m = m;
    sdata.lag = lag;
    sdata.semivariogram = semivariogram;

    gsl_multifit_function_fdf f;
    f.f = &model_f;
    f.df = nullptr;
    f.fdf = nullptr;
    f.n = m;
    f.p = p;
    f.params = &sdata;

    gsl_multifit_fdfsolver_set(s, &f, x);

    do{
        iter++;
        status = gsl_multifit_fdfsolver_iterate(s);
        if(status){
            break;
        }
        double new_scale_est = gsl_vector_get(s->x, 1);
        double new_range_est = gsl_vector_get(s->x, 0);
        if(new_scale_est < 1e-6){
            new_scale_est = 1e-6;
            gsl_vector_set(s->x, 1, new_scale_est);
        }
        if(new_range_est < 1e-6){
            new_range_est = 1e-6;
            gsl_vector_set(s->x, 0, new_range_est);
        }
        status = gsl_multifit_test_delta(s->dx, s->x, 1e-4, 1e-4);
    } while(status == GSL_CONTINUE && iter < max_iter);

    range_est = gsl_vector_get(s->x, 0);
    scale_est = gsl_vector_get(s->x, 1);

    gsl_vector_free(x);
    gsl_multifit_fdfsolver_free(s);
}

inline double
model_fitted(double lag, const double range, const double scale){
    return scale * (1.0 - exp(- (lag*lag) / (range*range)));
    // return scale * (1.0 - exp(- lag / range));
}

template <class T>
void update_kriging_weights(const double range, const double scale, int id, int d, std::vector<std::vector<AdjNode<T>>>& processed_adj_nodes, const std::vector<T>& positions){
    size_t m = processed_adj_nodes[id].size();

    gsl_vector * cov = gsl_vector_alloc(m+1);
    for(int i=0; i<m; i++){
        double h = distance_euclid(d, &positions[processed_adj_nodes[id][i].id*d], &positions[id*d]);
        double ci = model_fitted(h, range, scale);
        gsl_vector_set(cov, i, ci);
    }
    gsl_vector_set(cov, m, 1);

    gsl_matrix * Var = gsl_matrix_alloc(m+1, m+1);
    for(int i=0; i<m; i++){
        gsl_matrix_set(Var, i, m, 1);
        gsl_matrix_set(Var, m, i, 1);
        for(int j=0; j<m; j++){
            double h = distance_euclid(d, &positions[processed_adj_nodes[id][i].id*d], &positions[processed_adj_nodes[id][j].id*d]);
            double vij = model_fitted(h, range, scale);
            gsl_matrix_set(Var, i, j, vij);
        }
    }
    gsl_matrix_set(Var, m, m, 0);

    int signum = 0;
    gsl_permutation *p = gsl_permutation_alloc(m+1);
    gsl_linalg_LU_decomp(Var, p, &signum);
    gsl_matrix * inverse = gsl_matrix_alloc(m+1, m+1);
    gsl_linalg_LU_invert(Var, p, inverse);
    gsl_vector * lambda = gsl_vector_alloc(m+1);
    gsl_blas_dgemv(CblasNoTrans, 1.0, inverse, cov, 0, lambda);

    for(int i=0; i<m; i++){
        processed_adj_nodes[id][i].krg_weight = gsl_vector_get(lambda, i);
    }

    gsl_permutation_free(p);
    gsl_vector_free(cov);
    gsl_vector_free(lambda);
    gsl_matrix_free(Var);
    gsl_matrix_free(inverse);
}

}
#endif