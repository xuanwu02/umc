#ifndef _UMC_REGRESSION_HPP
#define _UMC_REGRESSION_HPP

#include <vector>
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_linalg.h>

double max_error = 0;
double squared_error = 0;
const double bound = 1e-15;

namespace UMC{


template<typename T>
gsl_matrix* generate_design_matrix(int n, int d, const std::vector<const T*>& input_X) {
    /* build a matrix X of size n * (d+1) with input_X's elements */
    gsl_matrix *X;
    X = gsl_matrix_alloc(n, d+1);
    for (int i = 0; i< n; i++) {
        gsl_matrix_set (X, i, 0, 1.0);
        for(int j=0; j<d; j++){
            gsl_matrix_set (X, i, j+1, input_X[i][j]);
        }
    }
    return X;
}

gsl_matrix* generate_multiplier2(int n, int d, gsl_matrix *& X) {
    /* calculate multiplier = (X'X)^(-1)X' to solve for regression coefficients
     * get pseudo inverse (X'X)^+ for non-singular X'X if det(X'X) < bound
     * X'X = U * Sigma * V',  S equals diag(Sigma)
     * (X'X)^+ = V * Sigma^(-1) * U'
     */
    gsl_matrix *XtX = gsl_matrix_alloc(d+1, d+1);
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, X, X, 0, XtX); // XtX = X'X
    gsl_matrix *XX = gsl_matrix_alloc(d+1, d+1);
    gsl_matrix_memcpy(XX, XtX);

    int signum = 0;
    gsl_permutation *p = gsl_permutation_alloc(d+1);
    gsl_linalg_LU_decomp(XtX, p, &signum); // store LU decomposition of X'X in XtX

    double det = gsl_linalg_LU_det(XtX, signum);
    gsl_matrix *prod = gsl_matrix_alloc(d+1, n);
    if(det >= bound){
        gsl_matrix *inverse = gsl_matrix_alloc(d+1, d+1);
        gsl_linalg_LU_invert(XtX, p, inverse); // inverse = (X'X)^(-1)
        gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, inverse, X, 0, prod); // prod = (X'X)^(-1)X' is d+1 by n
        gsl_matrix_free(inverse);
    }
    else{
        gsl_matrix *V = gsl_matrix_alloc(d+1, d+1);
        gsl_vector *S = gsl_vector_alloc(d+1);
        gsl_vector *work = gsl_vector_alloc(d+1);
        gsl_linalg_SV_decomp(XX, V, S, work); // store U in XX
        gsl_matrix *Sigma = gsl_matrix_alloc(d+1, d+1);
        gsl_matrix_set_zero(Sigma);
        for(int i=0; i<(d+1); i++){
            double si = gsl_vector_get(S, i);
            gsl_matrix_set(Sigma, i, i, si);
            if(si > bound) 
                gsl_matrix_set(Sigma, i, i, 1.0/si);
        }
        gsl_matrix *tmp = gsl_matrix_alloc(d+1, d+1);
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, V, Sigma, 0, tmp); // tmp = V * Sigma^(-1)
        gsl_matrix *inverse = gsl_matrix_alloc(d+1, d+1);
        gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, tmp, XX, 0, inverse); // inverse = tmp * U'
        gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, inverse, X, 0, prod); // prod = (X'X)^+X' is d+1 by n

        gsl_vector_free(S);
        gsl_vector_free(work);
        gsl_matrix_free(V);
        gsl_matrix_free(Sigma);
        gsl_matrix_free(tmp);
        gsl_matrix_free(inverse);
    }
    gsl_permutation_free(p);
    gsl_matrix_free(XtX);
    gsl_matrix_free(XX);
    return prod;
}

template<typename T>
void fit_and_quantize(int n, int d, const gsl_matrix* const& design_matrix, const gsl_matrix* const& multiplier, const std::vector<T>& input_y, 
                        SZ3::LinearQuantizer<T>& quantizer, std::vector<int>& quant_inds, SZ3::LinearQuantizer<T>& coeff_quantizer, 
                            std::vector<int>& coeff_quant_inds, std::vector<T>& coeff_prev) {
    gsl_vector *y, *c, *y_pred;
    y = gsl_vector_alloc(n);
    for(int i=0; i<n; i++){
        gsl_vector_set(y, i, input_y[i]);
    }
    c = gsl_vector_alloc(d+1);
    gsl_blas_dgemv(CblasNoTrans, 1.0, multiplier, y, 0, c); // c = multiplier * y 
    for (int i = 0; i < d+1; i++) {
        T cc = gsl_vector_get(c, i);                     
    	auto ind = coeff_quantizer.quantize_and_overwrite(cc, coeff_prev[i]);
    	coeff_quant_inds.push_back(ind);
    	coeff_prev[i] = cc;
    	gsl_vector_set(c, i, cc);
    }
    y_pred = gsl_vector_alloc(n);
    gsl_blas_dgemv(CblasNoTrans, 1.0, design_matrix, c, 0, y_pred);
    for (int i = 0; i < n; i++) {
        T decompressed = input_y[i];
        auto ind = quantizer.quantize_and_overwrite(decompressed, gsl_vector_get(y_pred, i));
        quant_inds.push_back(ind);
        double error = decompressed - input_y[i];
        if(fabs(error) > max_error) max_error = fabs(error);
        squared_error += error*error;
    }
    gsl_vector_free(y);
    gsl_vector_free(c);
    gsl_vector_free(y_pred);
}

template<typename T>
void est_and_recover(int n, int d, const gsl_matrix * design_matrix, SZ3::LinearQuantizer<T>& quantizer, 
                        const int *& quant_inds, SZ3::LinearQuantizer<T>& coeff_quantizer, const int *& coeff_quant_inds,
                            std::vector<T>& coeff_prev, std::vector<T>& output_Y){
    gsl_vector *y, *c;
    c = gsl_vector_alloc(d+1);
    for(int i=0; i<d+1; i++){
    	auto cc = coeff_quantizer.recover(coeff_prev[i], *(coeff_quant_inds ++));
    	coeff_prev[i] = cc;
    	gsl_vector_set(c, i, cc);
    }
    y = gsl_vector_alloc(n);
    gsl_blas_dgemv(CblasNoTrans, 1.0, design_matrix, c, 0, y);
    for(int i=0; i<n; i++){
        double y_est = gsl_vector_get(y, i);
        output_Y[i] = quantizer.recover(y_est, *(quant_inds ++));
    }
    gsl_vector_free(y);
    gsl_vector_free(c);
}

}
#endif