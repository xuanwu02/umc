#ifndef _UMC_COMPRESS_HPP
#define _UMC_COMPRESS_HPP

#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <cmath>
#include <time.h>
#include <SZ3/quantizer/IntegerQuantizer.hpp>
#include <SZ3/encoder/HuffmanEncoder.hpp>
#include <SZ3/lossless/Lossless_zstd.hpp>
#include <SZ3/api/sz.hpp>
#include "utils.hpp"
#include "reorder.hpp"
#include "partition.hpp"
#include "regression.hpp"
#include "adjacent_prediction.hpp"
#include "kriging_interpolation.hpp"
#include "set_decomposition.hpp"
#include "cp_preservation.hpp" 

namespace UMC{


// Compression Method 1: RBP
template <class T>
unsigned char *
umc_compress_prediction_by_regression(const T invalid_val, const T * data, size_t num_elements, int d, int elements_per_partition, const std::vector<gsl_matrix *>& design_matrix,
                                        const std::vector<gsl_matrix *>& multiplier, const std::vector<uint32_t>& num_part_nodes, const std::vector<uint32_t>& part_map_inv,
                                            size_t& compressed_size, double eb, bool use_abs_eb){
    std::vector<T> dec_data(num_elements);
    memcpy(dec_data.data(), data, num_elements*sizeof(T)); 
    std::vector<T> refined_data(dec_data);
    refined_data.erase(std::remove(refined_data.begin(), refined_data.end(), invalid_val), refined_data.end());
    auto max_val = *std::max_element(refined_data.begin(), refined_data.end());
    auto min_val = *std::min_element(refined_data.begin(), refined_data.end());
    auto valid_range = max_val - min_val;
    refined_data = std::vector<T>();
    if(!use_abs_eb){
        eb *= valid_range;
    }  
    const int quant_radius = 32768;
    const double reg_coeff_eb = 0.1;
    SZ3::LinearQuantizer<T> quantizer = SZ3::LinearQuantizer<T>(eb, quant_radius);
    SZ3::LinearQuantizer<T> coeff_quantizer = SZ3::LinearQuantizer<T>(reg_coeff_eb / (d+1), quant_radius);
    std::vector<int> quant_inds;
    std::vector<int> coeff_quant_inds;
    std::vector<T> reg_coeff(d+1, 0);
    squared_error = 0;
    max_error = 0;
    uint32_t offset = 0;
    for(int i=0; i<num_part_nodes.size(); i++){
        if(num_part_nodes[i]>0){
            std::vector<T> input_Y;
            for(int j=0; j<num_part_nodes[i]; j++){
                auto original_id = part_map_inv[j+offset];
                input_Y.push_back(data[original_id]);
            }
            fit_and_quantize(num_part_nodes[i], d, design_matrix[i], multiplier[i], input_Y, quantizer, quant_inds, coeff_quantizer, coeff_quant_inds, reg_coeff);
        }
        offset += num_part_nodes[i];
    }
    dec_data = std::vector<T>();
    unsigned char * compressed = (unsigned char *) malloc(num_elements * 3 * sizeof(T));
    unsigned char * compressed_data_pos = compressed;     
    write(eb, compressed_data_pos);
    size_t num_coeff_quant_inds = coeff_quant_inds.size();
    write(num_coeff_quant_inds, compressed_data_pos);
    size_t num_quant_inds = quant_inds.size();
    write(num_quant_inds, compressed_data_pos);
    coeff_quantizer.save(compressed_data_pos);
    quantizer.save(compressed_data_pos);
    auto coeff_encoder = SZ3::HuffmanEncoder<int>();
    coeff_encoder.preprocess_encode(coeff_quant_inds, 0);
    coeff_encoder.save(compressed_data_pos);
    coeff_encoder.encode(coeff_quant_inds, compressed_data_pos);
    coeff_encoder.postprocess_encode();
    auto encoder = SZ3::HuffmanEncoder<int>();
    encoder.preprocess_encode(quant_inds, 0);
    encoder.save(compressed_data_pos);
    encoder.encode(quant_inds, compressed_data_pos);
    encoder.postprocess_encode();
    auto lossless = SZ3::Lossless_zstd();
    std::cout << "size before lossless = " << compressed_data_pos - compressed << std::endl;
    unsigned char *lossless_data = lossless.compress(compressed, compressed_data_pos - compressed, compressed_size);
    lossless.postcompress_data(compressed);
    std::cout << "compressed_size = " << compressed_size << std::endl;
    std::cout << "compression_ratio = " << num_elements*sizeof(T) * 1.0 / compressed_size << std::endl;
    std::cout << "Max value = " << max_val << ", min value = " << min_val << ", valid range = " << valid_range << std::endl;
    std::cout << "Max error = " << max_error << std::endl;
    double mse = squared_error/num_elements;
    std::cout << "MSE = " << mse << std::endl;
    std::cout << "PSNR = " << 20 * log10((max_val - min_val) / sqrt(mse)) << std::endl;
    return lossless_data;
}

template <class T>
unsigned char *
umc_compress_prediction_by_regression(const std::vector<int32_t>& index_map, int d, int elements_per_partition, const std::string& data_file, const std::vector<gsl_matrix *>& design_matrix, 
                                        const std::vector<gsl_matrix *>& multiplier, const std::vector<uint32_t>& num_part_nodes, const std::vector<uint32_t>& part_map_inv, 
                                            size_t& compressed_size, double eb, bool use_abs_eb){
	struct timespec start, end, end1;
    int err = 0;
    double reorder_time = 0, compression_time = 0;;
    err = clock_gettime(CLOCK_REALTIME, &start);
    size_t num_elements = 0;
	auto data = readfile<T>(data_file.c_str(), num_elements);
    if(index_map[0] != -1) data = reorder(data, index_map, num_elements, 1);
    err = clock_gettime(CLOCK_REALTIME, &end);
    const T invalid_val = -99999.0;
    auto compressed = umc_compress_prediction_by_regression<T>(invalid_val, data.data(), num_elements, d, elements_per_partition, design_matrix, multiplier, num_part_nodes, part_map_inv, compressed_size, eb, use_abs_eb);
    err = clock_gettime(CLOCK_REALTIME, &end1);
    reorder_time = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
    printf("reorder time = %.6f\n", reorder_time);
    compression_time = (double)(end1.tv_sec - start.tv_sec) + (double)(end1.tv_nsec - start.tv_nsec)/(double)1000000000;
    printf("compression time = %.6f\n", compression_time);
    return compressed;
}

template <class T>
void umc_compress_prediction_by_regression(int reorder_opt, int d, int elements_per_partition, const std::string& position_file, const std::string& conn_file, int from_edge, const std::vector<std::string>& data_files, double eb, bool use_abs_eb){
	struct timespec start, end;
    int err = 0;
    err = clock_gettime(CLOCK_REALTIME, &start);
    double preprocessing_time = 0;
    size_t num = 0;
    auto positions = readfile<T>(position_file.c_str(), num);
    assert(num % d == 0);
    size_t num_elements = num / d;
    auto conn = readfile<int>(conn_file.c_str(), num);
    auto adj_list = generate_adjacent_list(num_elements, d, conn, from_edge);
    std::vector<int32_t> index_map(num_elements, -1);
    if(reorder_opt > 0){
        index_map = generate_reorder_index_map(num_elements, adj_list, reorder_opt);
        positions = reorder(positions, index_map, num_elements, d);
        conn = reorder_conn(conn, index_map, d);
        adj_list = generate_adjacent_list(num_elements, d, conn, from_edge);
    }
    conn = std::vector<int>();
	idx_t nparts = (num_elements - 1)/elements_per_partition + 1;
	auto part = (nparts == 1) ? std::vector<idx_t>(num_elements, 0) : create_partition(num_elements, nparts, adj_list);
	std::vector<uint32_t> part_map(num_elements, -1);
	std::vector<uint32_t> part_map_inv(num_elements, -1);
	std::vector<uint32_t> num_part_nodes(nparts, 0);
	for(int i=0; i<part.size(); i++){
	    num_part_nodes[part[i]] ++;
	}
	init_part_map(part, num_part_nodes, part_map, part_map_inv);
	std::vector<gsl_matrix *> design_matrix(nparts);
	std::vector<gsl_matrix *> multiplier(nparts);
	uint32_t offset = 0;
	for(int i=0; i<nparts; i++){
	    if(num_part_nodes[i]>0){
	        std::vector<const T*> input_X;
	        for(int j=0; j<num_part_nodes[i]; j++){
	            auto original_id = part_map_inv[j + offset];
	            input_X.push_back(&positions[d*original_id]);
	        }
	        gsl_matrix *X = generate_design_matrix(num_part_nodes[i], d, input_X);
	        design_matrix[i] = X;
	        multiplier[i] = generate_multiplier2(num_part_nodes[i], d, X);
	    }
	    offset += num_part_nodes[i];
	}
    err = clock_gettime(CLOCK_REALTIME, &end);
    preprocessing_time = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
    printf("preprocessing time = %.6f\n", preprocessing_time);
    for(int i=0; i<data_files.size(); i++){
    	size_t compressed_size = 0;
    	auto compressed = umc_compress_prediction_by_regression<T>(index_map, d, elements_per_partition, data_files[i], design_matrix, multiplier, num_part_nodes, part_map_inv, compressed_size, eb, use_abs_eb);
        writefile((data_files[i] + ".umc").c_str(), compressed, compressed_size);
    }
    for(int i=0; i<design_matrix.size(); i++){
    	gsl_matrix_free(design_matrix[i]);
    	gsl_matrix_free(multiplier[i]);
    }
}


// Compression Method 2: NBP-IDW
template <class T>
unsigned char *
umc_compress_prediction_by_adjacent_nodes_simple(const T invalid_val, const std::vector<int32_t>& index_map, const T * data, size_t num_elements,
                                                    const std::vector<std::vector<AdjNode<T>>>& processed_adj_nodes, int n_neighbors,
                                                        double eb, size_t& compressed_size, bool use_abs_eb){
    std::vector<T> dec_data(num_elements);
    memcpy(dec_data.data(), data, num_elements*sizeof(T)); 
    std::vector<T> refined_data(dec_data);
    refined_data.erase(std::remove(refined_data.begin(), refined_data.end(), invalid_val), refined_data.end());
    auto max_val = *std::max_element(refined_data.begin(), refined_data.end());
    auto min_val = *std::min_element(refined_data.begin(), refined_data.end());
    auto valid_range = max_val - min_val;
    refined_data = std::vector<T>();
    if(!use_abs_eb){
        eb *= valid_range;
    }    
    const int quant_radius = 32768;
    SZ3::LinearQuantizer<T> quantizer = SZ3::LinearQuantizer<T>(eb, quant_radius);
    std::vector<int> quant_inds;
    double squared_error = 0;
    double max_error = 0;
    for (int i=0; i<num_elements; i++){
        T pred = 0;
        for(int neighbor=0; neighbor<processed_adj_nodes[i].size(); neighbor++){
            int j = processed_adj_nodes[i][neighbor].id;
            pred += processed_adj_nodes[i][neighbor].weight * dec_data[j];
        }
        auto quant_ind = quantizer.quantize_and_overwrite(dec_data[i], pred);
        {
            double error = dec_data[i] - data[i];
            if(fabs(error) > max_error) max_error = fabs(error);
            squared_error += error*error;               
        }
        quant_inds.push_back(quant_ind);
    }
    unsigned char * compressed = (unsigned char *) malloc(num_elements*2 * sizeof(T));
    unsigned char * compressed_data_pos = compressed;
    write(n_neighbors, compressed_data_pos);       
    write(eb, compressed_data_pos);
    size_t num_quant_inds = quant_inds.size();
    write(num_quant_inds, compressed_data_pos);
    quantizer.save(compressed_data_pos);
    auto encoder = SZ3::HuffmanEncoder<int>();
    encoder.preprocess_encode(quant_inds, 0);
    encoder.save(compressed_data_pos);
    encoder.encode(quant_inds, compressed_data_pos);
    encoder.postprocess_encode();
    auto lossless = SZ3::Lossless_zstd();
    std::cout << "size before lossless = " << compressed_data_pos - compressed << std::endl;
    unsigned char *lossless_data = lossless.compress(compressed, compressed_data_pos - compressed, compressed_size);
    lossless.postcompress_data(compressed);
    std::cout << "compressed_size = " << compressed_size << std::endl;
    std::cout << "compression_ratio = " << num_elements*sizeof(T) * 1.0 / compressed_size << std::endl;
    std::cout << "Max value = " << max_val << ", min value = " << min_val << ", valid range = " << valid_range << std::endl;
    std::cout << "Max error = " << max_error << std::endl;
    double mse = squared_error/num_elements;
    std::cout << "MSE = " << mse << std::endl;
    std::cout << "PSNR = " << 20 * log10((max_val - min_val) / sqrt(mse)) << std::endl;
    return lossless_data;
}

template <class T>
unsigned char *
umc_compress_prediction_by_adjacent_nodes_simple(const std::vector<int32_t>& index_map, const std::string& data_file, 
                                                    const std::vector<std::vector<AdjNode<T>>>& processed_adj_nodes, 
                                                        int n_neighbors, size_t& compressed_size, double eb, bool use_abs_eb){
    struct timespec start, end, end1;
    int err = 0;
    double reorder_time = 0, compression_time = 0;;
    err = clock_gettime(CLOCK_REALTIME, &start);
    size_t num_elements = 0;
    auto data = readfile<T>(data_file.c_str(), num_elements);
    assert(num_elements == processed_adj_nodes.size());
    if(index_map[0] != -1) data = reorder(data, index_map, num_elements, 1);  
    err = clock_gettime(CLOCK_REALTIME, &end);
    const T invalid_val = -99999.0;
    auto compressed = umc_compress_prediction_by_adjacent_nodes_simple<T>(invalid_val, index_map, data.data(), num_elements, processed_adj_nodes, n_neighbors, eb, compressed_size, use_abs_eb);
    err = clock_gettime(CLOCK_REALTIME, &end1);
    reorder_time = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
    printf("reorder time = %.6f\n", reorder_time);
    compression_time = (double)(end1.tv_sec - start.tv_sec) + (double)(end1.tv_nsec - start.tv_nsec)/(double)1000000000;
    printf("compression time = %.6f\n", compression_time);
    return compressed;
}

template <class T>
void umc_compress_prediction_by_adjacent_nodes_simple(int reorder_opt, const std::string& position_file, const std::string& conn_file, int from_edge, 
                                                        const std::vector<std::string>& data_files, int n_neighbors, int d, double eb, bool use_abs_eb){
    struct timespec start, end;
    int err = 0;
    double preprocessing_time = 0, _time = 0;
    err = clock_gettime(CLOCK_REALTIME, &start);
    size_t num = 0;
    auto positions = readfile<T>(position_file.c_str(), num);
    assert(num % d == 0);
    size_t num_elements = num / d;
    auto conn = readfile<int>(conn_file.c_str(), num);
    auto processed_adj_list = generate_processed_adjacent_list(num_elements, d, conn, from_edge);
    std::vector<int32_t> index_map(num_elements, -1);
    if(reorder_opt > 0){
        auto original_adj_list = generate_adjacent_list(num_elements, d, conn, from_edge);
        index_map = generate_reorder_index_map(num_elements, original_adj_list, reorder_opt);      
        positions = reorder(positions, index_map, num_elements, d);
        conn = reorder_conn(conn, index_map, d);
        processed_adj_list = generate_processed_adjacent_list(num_elements, d, conn, from_edge);
        original_adj_list = std::vector<std::set<int32_t>>();
    }
    auto processed_adj_nodes = generate_adjacent_neighbors(num_elements, d, n_neighbors, processed_adj_list, positions);
    err = clock_gettime(CLOCK_REALTIME, &end);
    preprocessing_time = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
    printf("preprocessing time = %.6f\n", preprocessing_time);
    for(int i=0; i<data_files.size(); i++){
        size_t compressed_size = 0;
        auto compressed = umc_compress_prediction_by_adjacent_nodes_simple<T>(index_map, data_files[i], processed_adj_nodes, n_neighbors, compressed_size, eb, use_abs_eb);
        writefile((data_files[i] + ".umc").c_str(), compressed, compressed_size);
    }
}


// Compression Method 3: NBP-KW
template <class T>
unsigned char *
umc_compress_prediction_by_adjacent_nodes_kriging(const T invalid_val, int timestep, int batch_size, int d, const T * data, size_t num_elements, const std::vector<T>& positions,
                                                    std::vector<std::vector<AdjNode<T>>>& processed_adj_nodes, int min_neighbors, int max_iter, size_t& compressed_size, double eb, bool use_abs_eb){
    std::vector<T> dec_data(num_elements);
    memcpy(dec_data.data(), data, num_elements*sizeof(T)); 
    std::vector<T> refined_data(dec_data);
    refined_data.erase(std::remove(refined_data.begin(), refined_data.end(), invalid_val), refined_data.end());
    auto max_val = *std::max_element(refined_data.begin(), refined_data.end());
    auto min_val = *std::min_element(refined_data.begin(), refined_data.end());
    auto valid_range = max_val - min_val;
    refined_data = std::vector<T>();
    if(!use_abs_eb){
        eb *= valid_range;
    }
    const int quant_radius = 32768;
    SZ3::LinearQuantizer<T> quantizer = SZ3::LinearQuantizer<T>(eb, quant_radius);
    std::vector<int> quant_inds;
    double squared_error = 0;
    double max_error = 0;
    int krg_count = 0;
    bool renew = (timestep % batch_size == 0);
    for(int i=0; i<num_elements; i++){
        T pred = 0;
        double denominator = 0;
        if(processed_adj_nodes[i].size() < min_neighbors){
            for(int neighbor=0; neighbor<processed_adj_nodes[i].size(); neighbor++){
                int j = processed_adj_nodes[i][neighbor].id;
                pred += processed_adj_nodes[i][neighbor].weight * dec_data[j];
                denominator += processed_adj_nodes[i][neighbor].weight;
            }
            if(denominator > 0) pred /= denominator;         
        }else{
            krg_count ++;
            if(renew){      
                std::vector<double> spatial_lag;
                std::vector<double> observed_semivariogram;
                for(int j=0; j<processed_adj_nodes[i].size(); j++){
                    int id_j = processed_adj_nodes[i][j].id;
                    for(int k=0; k<processed_adj_nodes[i].size(); k++){
                        if(k > j){
                            int id_k = processed_adj_nodes[i][k].id;
                            double diff_jk = dec_data[id_j] - dec_data[id_k];
                            observed_semivariogram.push_back(diff_jk * diff_jk / 2.0);
                            spatial_lag.push_back(distance_euclid(d, &positions[id_j*d], &positions[id_k*d]));
                        }
                    }
                }
                double scale_est = 4.0;
                double range_est = *std::max_element(spatial_lag.begin(), spatial_lag.end()) / sqrt(3.0);
                fit_model(spatial_lag, observed_semivariogram, range_est, scale_est, max_iter);
                update_kriging_weights(range_est, scale_est, i, d, processed_adj_nodes, positions);
            }
            for(int j=0; j<processed_adj_nodes[i].size(); j++){
                pred += processed_adj_nodes[i][j].krg_weight * dec_data[processed_adj_nodes[i][j].id];
            }
        }
        auto quant_ind = quantizer.quantize_and_overwrite(dec_data[i], pred); 
        {
            double error = dec_data[i] - data[i];
            if(fabs(error) > max_error) max_error = fabs(error);
            squared_error += error*error;   
        }
        quant_inds.push_back(quant_ind);       
    }
    std::cout << "Kriging portion = " << krg_count << " / " << num_elements << " = " << (double)krg_count/num_elements << std::endl;
    unsigned char * compressed = (unsigned char *) malloc(num_elements * 2 * sizeof(T));
    unsigned char * compressed_data_pos = compressed;
    write(min_neighbors, compressed_data_pos);
    write(max_iter, compressed_data_pos);       
    write(eb, compressed_data_pos);
    size_t num_quant_inds = quant_inds.size();
    write(num_quant_inds, compressed_data_pos);
    quantizer.save(compressed_data_pos);
    auto encoder = SZ3::HuffmanEncoder<int>();
    encoder.preprocess_encode(quant_inds, 0);
    encoder.save(compressed_data_pos);
    encoder.encode(quant_inds, compressed_data_pos);
    encoder.postprocess_encode();
    auto lossless = SZ3::Lossless_zstd();
    std::cout << "size before lossless = " << compressed_data_pos - compressed << std::endl;
    unsigned char *lossless_data = lossless.compress(compressed, compressed_data_pos - compressed, compressed_size);
    lossless.postcompress_data(compressed);
    std::cout << "compressed_size = " << compressed_size << std::endl;
    std::cout << "compression_ratio = " << num_elements*sizeof(T) * 1.0 / compressed_size << std::endl;
    std::cout << "Max value = " << max_val << ", min value = " << min_val << ", valid range = " << valid_range << std::endl;
    std::cout << "Max error = " << max_error << std::endl;
    double mse = squared_error/num_elements;
    std::cout << "MSE = " << mse << std::endl;
    std::cout << "PSNR = " << 20 * log10((max_val - min_val) / sqrt(mse)) << std::endl;
    return lossless_data;
}

template <class T>
unsigned char *
umc_compress_prediction_by_adjacent_nodes_kriging(int timestep, int batch_size, int d, const std::vector<int32_t>& index_map, const std::string& data_file, const std::vector<T>& positions,
                                                    std::vector<std::vector<AdjNode<T>>>& processed_adj_nodes, int min_neighbors, int max_iter, size_t& compressed_size, double eb, bool use_abs_eb){
    struct timespec start, end, end1;
    int err = 0;
    double reorder_time = 0, compression_time = 0;;
    err = clock_gettime(CLOCK_REALTIME, &start);
    size_t num_elements = 0;
    auto data = readfile<T>(data_file.c_str(), num_elements);
    assert(num_elements == processed_adj_nodes.size());
    const T invalid_val = -99999.0;
    if(index_map[0] != -1) data = reorder(data, index_map, num_elements, 1);  
    err = clock_gettime(CLOCK_REALTIME, &end);
    auto compressed = umc_compress_prediction_by_adjacent_nodes_kriging<T>(invalid_val, timestep, batch_size, d, data.data(), num_elements, positions, processed_adj_nodes, min_neighbors, max_iter, compressed_size, eb, use_abs_eb);
    err = clock_gettime(CLOCK_REALTIME, &end1);
    reorder_time += (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
    printf("reorder time = %.6f\n", reorder_time);
    compression_time = (double)(end1.tv_sec - start.tv_sec) + (double)(end1.tv_nsec - start.tv_nsec)/(double)1000000000;
    printf("compression time = %.6f\n", compression_time);
    return compressed;
}

template <class T>
void umc_compress_prediction_by_adjacent_nodes_kriging(int reorder_opt, const std::string& position_file, const std::string& conn_file, int from_edge, 
                                                            const std::vector<std::string>& data_files, int min_neighbors, int max_iter, int batch_size,
                                                                int d, double eb, bool use_abs_eb){
    struct timespec start, end;
    int err = 0;
    err = clock_gettime(CLOCK_REALTIME, &start);
    double preprocessing_time = 0;
    size_t num = 0;
    auto positions = readfile<T>(position_file.c_str(), num);
    assert(num % d == 0);
    size_t num_elements = num / d;
    printf("num_elements = %zu\n", num_elements);
    auto conn = readfile<int>(conn_file.c_str(), num);
    auto processed_adj_list = generate_processed_adjacent_list(num_elements, d, conn, from_edge);
    std::vector<int32_t> index_map(num_elements, -1);
    if(reorder_opt > 0){
        auto original_adj_list = generate_adjacent_list(num_elements, d, conn, from_edge);
        index_map = generate_reorder_index_map(num_elements, original_adj_list, reorder_opt);
        positions = reorder(positions, index_map, num_elements, d);
        conn = reorder_conn(conn, index_map, d);
        processed_adj_list = generate_processed_adjacent_list(num_elements, d, conn, from_edge);
        original_adj_list = std::vector<std::set<int32_t>>();
    }
    auto processed_adj_nodes = generate_adjacent_neighbors2(num_elements, d, 4, processed_adj_list, positions);
    err = clock_gettime(CLOCK_REALTIME, &end);
    preprocessing_time = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
    printf("preprocessing time = %.6f\n", preprocessing_time);
    for(int i=0; i<data_files.size(); i++){
        size_t compressed_size = 0;
        auto compressed = umc_compress_prediction_by_adjacent_nodes_kriging<T>(i, batch_size, d, index_map, data_files[i], positions, processed_adj_nodes, min_neighbors, max_iter, compressed_size, eb, use_abs_eb);
        writefile((data_files[i] + ".umc").c_str(), compressed, compressed_size);
    }
    
}


// Compression Method 4: NBP-IDW-INP
template <class T>
unsigned char *
umc_compress_prediction_by_adjacent_nodes_dynamic(const T invalid_val, const std::vector<std::set<int32_t>>& original_adj_list, int timestep, int batch_size, int d, const T * data, size_t num_elements, 
                                                    const std::vector<T>& positions, int n_neighbors, const std::vector<std::vector<AdjNode<T>>>& ref_processed_adj_nodes,
                                                        std::vector<std::vector<AdjNode<T>>>& current_processed_adj_nodes, unsigned char *& is_invalid,
                                                            size_t& compressed_size, double eb, bool use_abs_eb){
    std::vector<T> dec_data(num_elements);
    memcpy(dec_data.data(), data, num_elements*sizeof(T)); 
    std::vector<T> refined_data(dec_data); 
    refined_data.erase(std::remove(refined_data.begin(), refined_data.end(), invalid_val), refined_data.end());
    auto max_val = *std::max_element(refined_data.begin(), refined_data.end());
    auto min_val = *std::min_element(refined_data.begin(), refined_data.end());
    auto valid_range = max_val - min_val;
    refined_data = std::vector<T>();
    if(!use_abs_eb){
        eb *= valid_range;
    }
    const int quant_radius = 32768;
    SZ3::LinearQuantizer<T> quantizer = SZ3::LinearQuantizer<T>(eb, quant_radius);
    std::vector<int> quant_inds;
    double squared_error = 0;
    double max_error = 0; 
    bool renew = (timestep % batch_size == 0);
    unsigned char * changed_invalid = (unsigned char *) malloc(num_elements * sizeof(unsigned char));
    if(renew){        
        for(int i=0; i<num_elements; i++){
            unsigned char curr_invalid = (data[i] == invalid_val);
            if(is_invalid[i] == curr_invalid) changed_invalid[i] = 0;
            else{
                changed_invalid[i] = 1;
                is_invalid[i] = curr_invalid;
            }
        }
        update_processed_neighbors_of_changes_nodes(num_elements, original_adj_list, ref_processed_adj_nodes, changed_invalid, is_invalid, current_processed_adj_nodes);
    }
    for (int i=0; i<num_elements; i++){
        T pred = 0;
        if(is_invalid[i]) pred = invalid_val;
        else{
            double denominator = 0;
            int num = (n_neighbors <= current_processed_adj_nodes[i].size()) ? n_neighbors : current_processed_adj_nodes[i].size();
            for(int neighbor=0; neighbor<num; neighbor++){
                int j = current_processed_adj_nodes[i][neighbor].id;
                double wt = current_processed_adj_nodes[i][neighbor].weight;
                pred += wt * dec_data[j];
                denominator += wt;
            }
            if(denominator > 0) pred /= denominator;
        }
        auto quant_ind = quantizer.quantize_and_overwrite(dec_data[i], pred);
        {
            double error = dec_data[i] - data[i];
            if(fabs(error) > max_error) max_error = fabs(error);
            squared_error += error*error;                   
        }
        quant_inds.push_back(quant_ind);
    }
    unsigned char * compressed = (unsigned char *) malloc(num_elements * 2 * sizeof(T));
    unsigned char * compressed_data_pos = compressed;
    write(invalid_val, compressed_data_pos);
    write(n_neighbors, compressed_data_pos);       
    write(eb, compressed_data_pos);
    size_t num_quant_inds = quant_inds.size();
    write(num_quant_inds, compressed_data_pos);
    quantizer.save(compressed_data_pos);
    if(renew) convertIntArray2ByteArray_fast_1b_to_result_sz(changed_invalid, num_elements, compressed_data_pos);
    auto encoder = SZ3::HuffmanEncoder<int>();
    encoder.preprocess_encode(quant_inds, 0);
    encoder.save(compressed_data_pos);
    encoder.encode(quant_inds, compressed_data_pos);
    encoder.postprocess_encode();
    auto lossless = SZ3::Lossless_zstd();
    std::cout << "size before lossless = " << compressed_data_pos - compressed << std::endl;
    unsigned char *lossless_data = lossless.compress(compressed, compressed_data_pos - compressed, compressed_size);
    lossless.postcompress_data(compressed);
    std::cout << "compressed_size = " << compressed_size << std::endl;
    std::cout << "compression_ratio = " << num_elements*sizeof(T) * 1.0 / compressed_size << std::endl;
    std::cout << "Max value = " << max_val << ", min value = " << min_val << ", valid range = " << valid_range << std::endl;
    std::cout << "Max error = " << max_error << std::endl;
    double mse = squared_error/num_elements;
    std::cout << "MSE = " << mse << std::endl;
    std::cout << "PSNR = " << 20 * log10((max_val - min_val) / sqrt(mse)) << std::endl;
    free(changed_invalid);
    return lossless_data;   
}

template <class T>
unsigned char *
umc_compress_prediction_by_adjacent_nodes_dynamic(const std::vector<std::set<int32_t>>& original_adj_list, int timestep, int batch_size, int d, const std::vector<int32_t>& index_map,
                                                    const std::string& data_file, const std::vector<T>& positions, int n_neighbors, const std::vector<std::vector<AdjNode<T>>>& ref_processed_adj_nodes, 
                                                        std::vector<std::vector<AdjNode<T>>>& current_processed_adj_nodes, unsigned char*& is_invalid, size_t& compressed_size, double eb, bool use_abs_eb){
    struct timespec start, end, end1;
    int err = 0;
    double reorder_time = 0, compression_time = 0;;
    err = clock_gettime(CLOCK_REALTIME, &start);
    size_t num_elements = 0;
    auto data = readfile<T>(data_file.c_str(), num_elements);
    const T invalid_val = -99999.0;
    if(index_map[0] != -1) data = reorder(data, index_map, num_elements, 1); 
    err = clock_gettime(CLOCK_REALTIME, &end);
    auto compressed = umc_compress_prediction_by_adjacent_nodes_dynamic<T>(invalid_val, original_adj_list, timestep, batch_size, d, data.data(), num_elements, positions, n_neighbors, ref_processed_adj_nodes, current_processed_adj_nodes, is_invalid, compressed_size, eb, use_abs_eb);
    err = clock_gettime(CLOCK_REALTIME, &end1);
    reorder_time = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
    printf("reorder time = %.6f\n", reorder_time);
    compression_time = (double)(end1.tv_sec - start.tv_sec) + (double)(end1.tv_nsec - start.tv_nsec)/(double)1000000000;
    printf("compression time = %.6f\n", compression_time);
    return compressed;    
}

template <class T>
void umc_compress_prediction_by_adjacent_nodes_dynamic(int reorder_opt, const std::string& position_file, const std::string& conn_file, int from_edge,
                                                            const std::vector<std::string>& data_files, int n_neighbors, int batch_size, int d, double eb, bool use_abs_eb){
    struct timespec start, end;
    int err = 0;
    err = clock_gettime(CLOCK_REALTIME, &start);
    double preprocessing_time = 0;
    size_t num = 0;
    auto positions = readfile<T>(position_file.c_str(), num);
    assert(num % d == 0);
    size_t num_elements = num / d;
    printf("num_elements = %zu\n", num_elements);
    auto conn = readfile<int>(conn_file.c_str(), num);
    auto processed_adj_list = generate_processed_adjacent_list(num_elements, d, conn, from_edge);
    std::vector<int32_t> index_map(num_elements, -1);
    auto original_adj_list = generate_adjacent_list(num_elements, d, conn, from_edge);
    if(reorder_opt > 0){
        index_map = generate_reorder_index_map(num_elements, original_adj_list, reorder_opt);
        positions = reorder(positions, index_map, num_elements, d);
        conn = reorder_conn(conn, index_map, d);
        processed_adj_list = generate_processed_adjacent_list(num_elements, d, conn, from_edge);
        original_adj_list = generate_adjacent_list(num_elements, d, conn, from_edge);
    }
    auto ref_processed_adj_nodes = generate_adjacent_neighbors2(num_elements, d, processed_adj_list, positions);
    std::vector<std::vector<AdjNode<T>>> current_processed_adj_nodes(ref_processed_adj_nodes);
    unsigned char * is_invalid = (unsigned char *)malloc(num_elements * sizeof(unsigned char));
    memset(is_invalid, 0, num_elements * sizeof(unsigned char));
    err = clock_gettime(CLOCK_REALTIME, &end);
    preprocessing_time = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
    printf("preprocessing time = %.6f\n", preprocessing_time);
    for(int i=0; i<data_files.size(); i++){
        size_t compressed_size = 0;
        auto compressed = umc_compress_prediction_by_adjacent_nodes_dynamic<T>(original_adj_list, i, batch_size, d, index_map, data_files[i], positions, n_neighbors, ref_processed_adj_nodes, current_processed_adj_nodes, is_invalid, compressed_size, eb, use_abs_eb);
        writefile((data_files[i] + ".umc").c_str(), compressed, compressed_size);
    }
    free(is_invalid);
}


// Compression Method 5: ADP-1
template <class T>
unsigned char * umc_compress_blockwise_adaptive1(const T invalid_val, int d, size_t num_elements, const int n_neighbors, const std::vector<uint32_t>& num_part_nodes, 
                                                    const std::vector<uint32_t>& part_map_inv, const std::vector<std::vector<AdjNode<T>>>& processed_adj_nodes,
                                                        const std::vector<T>& data, size_t& compressed_size, double eb, bool use_abs_eb){ 
    std::vector<T> refined_data(data);
    refined_data.erase(std::remove(refined_data.begin(), refined_data.end(), invalid_val), refined_data.end());
    auto max_val = *std::max_element(refined_data.begin(), refined_data.end());
    auto min_val = *std::min_element(refined_data.begin(), refined_data.end());
    auto valid_range = max_val - min_val;
    refined_data = std::vector<T>();
    if(!use_abs_eb){
        eb *= valid_range;
    }
    double squared_error = 0, max_error = 0;
    const int capacity = 65536;
    const int quant_radius = (capacity >> 1);
    std::vector<int> quant_inds;
    size_t nparts = num_part_nodes.size();
    std::vector<T> dec_data(num_elements, 0);
    std::vector<T> dec_data1(data);
    SZ3::LinearQuantizer<T> quantizer = SZ3::LinearQuantizer<T>(eb, quant_radius);
    std::vector<unsigned char> pred_method(nparts, 0);
    uint32_t offset = 0;
    int count = 0;
    for(int i=0; i<nparts; i++){
        auto block_size = num_part_nodes[i];
        if(block_size == 0) continue;
        // SZ3
        std::vector<T> slice(block_size);
        std::copy(data.begin() + offset, data.begin() + offset + block_size, slice.begin());
        SZ3::Config conf = SZ3::Config(block_size);
        conf.absErrorBound = eb;
        conf.cmprAlgo = SZ3::ALGO_INTERP;
        // conf.cmprAlgo = SZ3::ALGO_LORENZO_REG;
        // conf.lorenzo = true;
        // conf.lorenzo2 = false;
        // conf.regression = false;
        // conf.regression2 = false;
        size_t sz_compressed_size = 0;
        char * sz_compressed = SZ_compress<float>(conf, slice.data(), sz_compressed_size);
        double sz3_cr = block_size * 1.0 * sizeof(float) / sz_compressed_size;
        free(sz_compressed);
        // Adj
        std::vector<int> block_quant_inds;
        std::vector<T> block_unpred_data; 
        for(int j=0; j<block_size; j++){
            auto original_id = part_map_inv[offset + j];
            T pred = 0;
            for(int k=0; k<processed_adj_nodes[original_id].size(); k++){
                int id_k = processed_adj_nodes[original_id][k].id;
                pred += processed_adj_nodes[original_id][k].weight * dec_data1[id_k];
            }
            T curr_data = data[original_id];
            T diff = curr_data - pred;
            T quant_diff = fabs(diff) / eb + 1;
            int quant_index = 0;
            bool unpred_flag = false;
            if(quant_diff < capacity){
                quant_diff = (diff > 0) ? quant_diff : -quant_diff;
                quant_index = (int)(quant_diff/2) + quant_radius;
                T dec_value = pred + 2 * (quant_index - quant_radius) * eb;
                T abs_error = fabs(dec_value - curr_data);
                if(abs_error >= eb){
                    unpred_flag = true;
                }else{
                    block_quant_inds.push_back(quant_index);
                    dec_data1[original_id] = dec_value;
                }
            }else{
                unpred_flag = true;
            }
            if(unpred_flag){
                block_quant_inds.push_back(0);
                block_unpred_data.push_back(curr_data);
                dec_data1[original_id] = curr_data;
            }
        }
        unsigned char * compressed = (unsigned char *) malloc(2*block_size*sizeof(T));
        unsigned char * compressed_data_pos = compressed;
        write(eb, compressed_data_pos);
        size_t unpred_data_count = block_unpred_data.size();
        write(unpred_data_count, compressed_data_pos);
        size_t num_quant_inds = block_quant_inds.size();
        write(num_quant_inds, compressed_data_pos);
        write_array_to_dst(compressed_data_pos, block_unpred_data.data(), unpred_data_count);
        Huffman_encode_tree_and_data(2*capacity, block_quant_inds.data(), num_quant_inds, compressed_data_pos);
        auto lossless = SZ3::Lossless_zstd();
        size_t adj_compressed_size = 0;
        unsigned char *block_lossless_data = lossless.compress(compressed, compressed_data_pos - compressed, adj_compressed_size);
        double adj_cr = block_size * sizeof(T) * 1.0 / adj_compressed_size;
        lossless.postcompress_data(compressed);
        pred_method[i] = (adj_cr < sz3_cr) ? 0 : 1;
        count += pred_method[i];
        if(pred_method[i] == 1){
            quantizer.set_eb(eb * 0.5);
            for(int j=0; j<block_size; j++){
                auto original_id = part_map_inv[offset+j];
                T pred = 0;
                for(int k=0; k<processed_adj_nodes[original_id].size(); k++){
                    int id = processed_adj_nodes[original_id][k].id;
                    pred += processed_adj_nodes[original_id][k].weight * dec_data[id];
                }
                dec_data[original_id] = data[original_id];
                auto quant_ind = quantizer.quantize_and_overwrite(dec_data[original_id], pred);
                {
                    double error = dec_data[original_id] - data[original_id];
                    if(fabs(error) > max_error) max_error = fabs(error);
                    squared_error += error*error;          		
                }
                quant_inds.push_back(quant_ind);
            } 
        }else{
            quantizer.set_eb(eb);
            // use interpolation
            uint interpolation_level = (uint) ceil(log2(block_size));
            double eb_ratio = 0.5;
            // quantize data 0
            auto curr_id = part_map_inv[offset];
            dec_data[curr_id] = data[curr_id];
            auto quant_ind = quantizer.quantize_and_overwrite(dec_data[curr_id], 0);
            quant_inds.push_back(quant_ind);
            {
                double error = dec_data[curr_id] - data[curr_id];
                if(fabs(error) > max_error) max_error = fabs(error);
                squared_error += error*error;                   
            }
            fflush(stdout);
            for(uint level = interpolation_level; level > 0 && level <= interpolation_level; level--){
                if(level >= 3){
                    quantizer.set_eb(eb * eb_ratio);
                }else{
                    quantizer.set_eb(eb);
                }
                size_t stride = 1U << (level - 1);
                size_t level_num = (block_size - 1) / stride + 1;
                if(level_num < 5){
                    // linear interp
                    for(size_t i = 1; i + 1 < level_num; i += 2){
                        auto curr_id = part_map_inv[offset+i*stride];
                        auto prev_id = part_map_inv[offset+(i-1)*stride];
                        auto next_id = part_map_inv[offset+(i+1)*stride];
                        T pred = (dec_data[prev_id] + dec_data[next_id]) * 0.5;
                        dec_data[curr_id] = data[curr_id];
                        auto quant_ind = quantizer.quantize_and_overwrite(dec_data[curr_id], pred);
                        quant_inds.push_back(quant_ind);
                        {
                            double error = dec_data[curr_id] - data[curr_id];
                            if(fabs(error) > max_error) max_error = fabs(error);
                            squared_error += error*error;                   
                        }
                    }
                    if(level_num % 2 == 0){
                        // Lorenzo
                        size_t i = (level_num - 1);
                        auto curr_id = part_map_inv[offset+i*stride];
                        auto prev_id = part_map_inv[offset+(i-1)*stride];
                        T pred = dec_data[prev_id];
                        dec_data[curr_id] = data[curr_id];
                        auto quant_ind = quantizer.quantize_and_overwrite(dec_data[curr_id], pred);
                        quant_inds.push_back(quant_ind);
                        {
                            double error = dec_data[curr_id] - data[curr_id];
                            if(fabs(error) > max_error) max_error = fabs(error);
                            squared_error += error*error;                   
                        }
                    }                
                }else{
                    // cubic interp
                    size_t i = 1;
                    auto curr_id = part_map_inv[offset+i*stride];
                    auto prev_id = part_map_inv[offset+(i-1)*stride];
                    auto next_id = part_map_inv[offset+(i+1)*stride];
                    // linear interp for the first data
                    T pred = (dec_data[prev_id] + dec_data[next_id]) * 0.5;
                    dec_data[curr_id] = data[curr_id];
                    auto quant_ind = quantizer.quantize_and_overwrite(dec_data[curr_id], pred);
                    quant_inds.push_back(quant_ind);
                    {
                        double error = dec_data[curr_id] - data[curr_id];
                        if(fabs(error) > max_error) max_error = fabs(error);
                        squared_error += error*error;                   
                    }

                    for(size_t i = 3; i + 3 < level_num; i += 2){
                        auto curr_id = part_map_inv[offset+i*stride];
                        auto prev_id = part_map_inv[offset+(i-1)*stride];
                        auto next_id = part_map_inv[offset+(i+1)*stride];
                        auto prev_prev_id = part_map_inv[offset+(i-3)*stride];
                        auto next_next_id = part_map_inv[offset+(i+3)*stride];
                        T pred = SZ3::interp_cubic(dec_data[prev_prev_id], dec_data[prev_id], dec_data[next_id], dec_data[next_next_id]);
                        dec_data[curr_id] = data[curr_id];
                        auto quant_ind = quantizer.quantize_and_overwrite(dec_data[curr_id], pred);
                        quant_inds.push_back(quant_ind);
                        {
                            double error = dec_data[curr_id] - data[curr_id];
                            if(fabs(error) > max_error) max_error = fabs(error);
                            squared_error += error*error;                   
                        }
                    }
                    if(level_num % 2 == 0){
                        // linear
                        size_t i = (level_num - 3);
                        auto curr_id = part_map_inv[offset+i*stride];
                        auto prev_id = part_map_inv[offset+(i-1)*stride];
                        auto next_id = part_map_inv[offset+(i+1)*stride];
                        T pred = (dec_data[prev_id] + dec_data[next_id]) * 0.5;
                        dec_data[curr_id] = data[curr_id];
                        auto quant_ind = quantizer.quantize_and_overwrite(dec_data[curr_id], pred);
                        quant_inds.push_back(quant_ind);
                        {
                            double error = dec_data[curr_id] - data[curr_id];
                            if(fabs(error) > max_error) max_error = fabs(error);
                            squared_error += error*error;                   
                        }
                        // lorenzo
                        i += 2;
                        curr_id = part_map_inv[offset+i*stride];
                        prev_id = part_map_inv[offset+(i-1)*stride];
                        pred = dec_data[prev_id];
                        dec_data[curr_id] = data[curr_id];
                        quant_ind = quantizer.quantize_and_overwrite(dec_data[curr_id], pred);
                        quant_inds.push_back(quant_ind);
                        {
                            double error = dec_data[curr_id] - data[curr_id];
                            if(fabs(error) > max_error) max_error = fabs(error);
                            squared_error += error*error;                   
                        }
                    }else{
                        // linear
                        size_t i = (level_num - 2);
                        auto curr_id = part_map_inv[offset+i*stride];
                        auto prev_id = part_map_inv[offset+(i-1)*stride];
                        auto next_id = part_map_inv[offset+(i+1)*stride];
                        T pred = (dec_data[prev_id] + dec_data[next_id]) * 0.5;
                        dec_data[curr_id] = data[curr_id];
                        auto quant_ind = quantizer.quantize_and_overwrite(dec_data[curr_id], pred);
                        quant_inds.push_back(quant_ind);
                        {
                            double error = dec_data[curr_id] - data[curr_id];
                            if(fabs(error) > max_error) max_error = fabs(error);
                            squared_error += error*error;                   
                        }
                    }
                }
            }             
        }
        offset += block_size;
    }
    std::cout << "SZ3 percent = " << 1 - count * 1.0 / nparts << std::endl;
    unsigned char * compressed = (unsigned char *) malloc(num_elements*3 * sizeof(T));
    unsigned char * compressed_data_pos = compressed;
    write(n_neighbors, compressed_data_pos);       
    write(eb, compressed_data_pos);
    size_t num_quant_inds = quant_inds.size();
    write(num_quant_inds, compressed_data_pos);
    quantizer.save(compressed_data_pos);
    convertIntArray2ByteArray_fast_1b_to_result_sz(pred_method.data(), pred_method.size(), compressed_data_pos);
    auto encoder = SZ3::HuffmanEncoder<int>();
    encoder.preprocess_encode(quant_inds, 0);
    encoder.save(compressed_data_pos);
    encoder.encode(quant_inds, compressed_data_pos);
    encoder.postprocess_encode();
    auto lossless = SZ3::Lossless_zstd();
    std::cout << "size before lossless = " << compressed_data_pos - compressed << std::endl;
    unsigned char *lossless_data = lossless.compress(compressed, compressed_data_pos - compressed, compressed_size);
    lossless.postcompress_data(compressed);
    std::cout << "compressed_size = " << compressed_size << std::endl;
    std::cout << "compression_ratio = " << num_elements*sizeof(T) * 1.0 / compressed_size << std::endl;
    std::cout << "Max value = " << max_val << ", min value = " << min_val << ", valid range = " << valid_range << std::endl;
    std::cout << "Max error = " << max_error << std::endl;
    double mse = squared_error/num_elements;
    std::cout << "MSE = " << mse << std::endl;
    std::cout << "PSNR = " << 20 * log10((max_val - min_val) / sqrt(mse)) << std::endl;
    return lossless_data;
}

template <class T>
unsigned char * umc_compress_blockwise_adaptive1(int d, const std::vector<int32_t>& index_map, const int n_neighbors, const std::string& data_file, const std::vector<uint32_t>& num_part_nodes, const std::vector<uint32_t>& part_map_inv,
                                const std::vector<std::vector<AdjNode<T>>>& processed_adj_nodes, size_t& compressed_size, double eb, bool use_abs_eb){
    struct timespec start, end, end1;
    int err = 0;
    double reorder_time = 0, compression_time = 0;;
    err = clock_gettime(CLOCK_REALTIME, &start);
    size_t num_elements = 0;
	auto data = readfile<T>(data_file.c_str(), num_elements);
    const T invalid_val = -99999.0;
    if(index_map[0] != -1) data = reorder(data, index_map, num_elements, 1);  
    err = clock_gettime(CLOCK_REALTIME, &end);
    auto compressed = umc_compress_blockwise_adaptive1<T>(invalid_val, d, num_elements, n_neighbors, num_part_nodes, part_map_inv, processed_adj_nodes, data, compressed_size, eb, use_abs_eb);
    err = clock_gettime(CLOCK_REALTIME, &end1);
    reorder_time = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
    printf("reorder time = %.6f\n", reorder_time);
    compression_time = (double)(end1.tv_sec - start.tv_sec) + (double)(end1.tv_nsec - start.tv_nsec)/(double)1000000000;
    printf("compression time = %.6f\n", compression_time);   
    return compressed; 
}

template <class T>
void umc_compress_blockwise_adaptive1(int reorder_opt, int d, const std::string& position_file, const std::string& conn_file, int from_edge, int n_neighbors,
                                        int elements_per_partition, const std::vector<std::string>& data_files, double eb, bool use_abs_eb){   
    struct timespec start, end;
    int err = 0;
    double preprocessing_time = 0; 
    err = clock_gettime(CLOCK_REALTIME, &start);
    size_t num = 0;
    auto positions = readfile<T>(position_file.c_str(), num);
    size_t num_elements = num / d;
    auto conn = readfile<int>(conn_file.c_str(), num);
    std::vector<int32_t> index_map(num_elements, -1);
    auto adj_list = generate_adjacent_list(num_elements, d, conn, from_edge);
    if(reorder_opt > 0){
        index_map = generate_reorder_index_map(num_elements, adj_list, reorder_opt);
        positions = reorder(positions, index_map, num_elements, d);
        conn = reorder_conn(conn, index_map, d);
        adj_list = generate_adjacent_list(num_elements, d, conn, from_edge);
    }
    idx_t nparts = (num_elements - 1)/elements_per_partition + 1;
    auto part = (nparts == 1) ? std::vector<idx_t>(num_elements, 0) : create_partition(num_elements, nparts, adj_list);
    std::vector<uint32_t> part_map(num_elements, -1);
    std::vector<uint32_t> part_map_inv(num_elements, -1);
    std::vector<uint32_t> num_part_nodes(nparts, 0);
    for(int i=0; i<part.size(); i++){
        num_part_nodes[part[i]] ++;
    }
    init_part_map(part, num_part_nodes, part_map, part_map_inv);
    auto part_map_new = blockDPFS(adj_list, part, part_map_inv, num_part_nodes);
    for(int i=0; i<num_elements; i++){
        part_map_inv[part_map_new[i]] = i;
    }
    auto regenerated_processed_adjacent_list = regenerate_processed_adjacent_list(num_elements, adj_list, part_map_new);
    auto processed_adj_nodes = generate_adjacent_neighbors(num_elements, d, n_neighbors, regenerated_processed_adjacent_list, positions);
    err = clock_gettime(CLOCK_REALTIME, &end);
    preprocessing_time = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
    printf("preprocessing time = %.6f\n", preprocessing_time);
    for(int i=0; i<data_files.size(); i++){
        size_t compressed_size = 0;
        auto compressed = umc_compress_blockwise_adaptive1<T>(d, index_map, n_neighbors, data_files[i], num_part_nodes, part_map_inv, processed_adj_nodes, compressed_size, eb, use_abs_eb);
        writefile((data_files[i] + ".umc").c_str(), compressed, compressed_size);
    }
}


// Compression Method 6: ADP-2
template <class T>
unsigned char * umc_compress_blockwise_adaptive2(const T invalid_val, int d, const int optimal_n_neighbors, const int optimal_reg_size, int blockwise_sample_size, const std::vector<T>& data,
                                                    const std::vector<int32_t>& sampleNodes_index, const std::vector<uint32_t>& num_part_nodes, const std::vector<uint32_t>& part_map_inv,
                                                        const std::vector<std::vector<AdjNode<T>>>& processed_adj_nodes, const std::vector<gsl_matrix *>& design_matrix, std::vector<gsl_matrix *>& multiplier,
                                                            const std::vector<T>& positions, size_t& compressed_size, double eb, bool use_abs_eb){
    size_t num_elements = data.size();
    std::vector<T> dec_data(data);
    std::vector<T> refined_data(data);
    refined_data.erase(std::remove(refined_data.begin(), refined_data.end(), invalid_val), refined_data.end());
    auto max_val = *std::max_element(refined_data.begin(), refined_data.end());
    auto min_val = *std::min_element(refined_data.begin(), refined_data.end());
    auto valid_range = max_val - min_val;
    refined_data = std::vector<T>();
    if(!use_abs_eb){
        eb *= valid_range;
    }
    const int quant_radius = 32768;
    const double reg_coeff_eb = 0.1;
    SZ3::LinearQuantizer<T> quantizer = SZ3::LinearQuantizer<T>(eb, quant_radius);
    SZ3::LinearQuantizer<T> tmp_coeff_quantizer = SZ3::LinearQuantizer<T>(reg_coeff_eb / (d+1), quant_radius);
    SZ3::LinearQuantizer<T> coeff_quantizer = SZ3::LinearQuantizer<T>(reg_coeff_eb / (d+1), quant_radius);
    // deal with each block
    size_t nparts = num_part_nodes.size();
    std::vector<unsigned char> pred_method(nparts, 0);
    std::vector<int> quant_inds, coeff_quant_inds;
    std::vector<T> coeff_prev(d+1, 0);
    double total_error = 0, max_error = -1;
    int regression_count = 0;
    uint32_t offset = 0;
    for(int i=0; i<nparts; i++){
        auto block_size = num_part_nodes[i];
        double reg_error = DBL_MAX, adj_error = DBL_MAX;
        if(block_size == 0) continue;
        reg_error = 0;
        adj_error = 0;
        std::vector<T> input_Y;
        gsl_vector *y = gsl_vector_alloc(block_size);
        // compute regression coefficient
        for(int j=0; j<block_size; j++){
            auto original_id = part_map_inv[offset+j];
            input_Y.push_back(data[original_id]);
            gsl_vector_set(y, j, data[original_id]);
        }
        gsl_vector *c = gsl_vector_alloc(d+1);
        gsl_blas_dgemv(CblasNoTrans, 1.0, multiplier[i], y, 0, c);
        std::vector<T> fitted_coeff(d+1), dec_coeff(d+1);
        // compute recovered coefficient
        for(int k=0; k<d+1; k++){
            fitted_coeff[k] = gsl_vector_get(c, k);
            dec_coeff[k] = fitted_coeff[k];
            auto coeff_quant_ind = tmp_coeff_quantizer.quantize_and_overwrite(dec_coeff[k], coeff_prev[k]);
            gsl_vector_set(c, k, dec_coeff[k]);
        }
        input_Y = std::vector<T>();
        // perform prediction on sampling data
        std::vector<const T*> sample_input_X;
        std::vector<T> sample_input_Y;
        gsl_vector *sample_y = gsl_vector_alloc(blockwise_sample_size);
        for(int j=0; j<blockwise_sample_size; j++){
            // NBP
            auto original_id = part_map_inv[offset + sampleNodes_index[j]];
            T adj_pred = 0;
            for(int k=0; k<processed_adj_nodes[original_id].size(); k++){
                int id = processed_adj_nodes[original_id][k].id;
                adj_pred += processed_adj_nodes[original_id][k].weight * data[id];
            }
            adj_error += fabs(data[original_id] - adj_pred);
            // RBP
            sample_input_X.push_back(&positions[d*original_id]);
            sample_input_Y.push_back(data[original_id]);
            gsl_vector_set(sample_y, j, data[original_id]);
        }
        gsl_matrix *sample_design_matrix = generate_design_matrix(blockwise_sample_size, d, sample_input_X);
        gsl_vector *sample_y_pred = gsl_vector_alloc(blockwise_sample_size);
        gsl_blas_dgemv(CblasNoTrans, 1.0, sample_design_matrix, c, 0, sample_y_pred);
        gsl_vector_sub(sample_y, sample_y_pred);
        for(int j=0; j<blockwise_sample_size; j++){
            reg_error += fabs(gsl_vector_get(sample_y, j));
        }
        pred_method[i] = adj_error <= reg_error ? 0 : 1;
        gsl_vector_free(sample_y);
        gsl_vector_free(sample_y_pred);
        gsl_matrix_free(sample_design_matrix);
        // compress current block
        if(pred_method[i] == 0){
            for(int j=0; j<block_size; j++){
                bool unpred_flag = false;
                auto original_id = part_map_inv[offset+j];
                T pred = 0;
                for(int k=0; k<processed_adj_nodes[original_id].size(); k++){
                    int id = processed_adj_nodes[original_id][k].id;
                    pred += processed_adj_nodes[original_id][k].weight * dec_data[id]; 
                }
                auto quant_ind = quantizer.quantize_and_overwrite(dec_data[original_id], pred);
                {
                    double error = dec_data[original_id] - data[original_id];
                    if(fabs(error) > max_error) max_error = fabs(error);
                    total_error += error*error;                 
                }
                quant_inds.push_back(quant_ind);
            }
        }else{
            regression_count ++;
            for(int k=0; k<d+1; k++){
                auto coeff_quant_ind = coeff_quantizer.quantize_and_overwrite(fitted_coeff[k], coeff_prev[k]);
                coeff_quant_inds.push_back(coeff_quant_ind);
            }
            coeff_prev = dec_coeff;
            gsl_vector *y_pred = gsl_vector_alloc(block_size);
            gsl_blas_dgemv(CblasNoTrans, 1.0, design_matrix[i], c, 0, y_pred);
            for(int j=0; j<block_size; j++){
                auto original_id = part_map_inv[offset+j];
                T pred = gsl_vector_get(y_pred, j);
                auto quant_ind = quantizer.quantize_and_overwrite(dec_data[original_id], pred);
                {
                    double error = dec_data[original_id] - data[original_id];
                    if(fabs(error) > max_error) max_error = fabs(error);
                    total_error += error*error;                 
                }
                quant_inds.push_back(quant_ind);
            }
            gsl_vector_free(y_pred);
        }
        gsl_vector_free(y);
        gsl_vector_free(c);
        offset += block_size;
    }
    std::cout << "# blocks = " << nparts << ", "<< "regression count = " << regression_count << ", portion = " << regression_count*1.0 / nparts << std::endl;
    dec_data = std::vector<T>();
    unsigned char * compressed = (unsigned char *) malloc(num_elements * 2 * sizeof(T));
    unsigned char * compressed_data_pos = compressed;
    write(optimal_n_neighbors, compressed_data_pos);
    write(optimal_reg_size, compressed_data_pos);     
    write(eb, compressed_data_pos);
    size_t num_coeff_quant_inds = coeff_quant_inds.size();
    write(num_coeff_quant_inds, compressed_data_pos);
    size_t num_quant_inds = quant_inds.size();
    write(num_quant_inds, compressed_data_pos);
    coeff_quantizer.save(compressed_data_pos);
    quantizer.save(compressed_data_pos);
    convertIntArray2ByteArray_fast_1b_to_result_sz(pred_method.data(), pred_method.size(), compressed_data_pos);
    if(num_coeff_quant_inds > 0){
        auto coeff_encoder = SZ3::HuffmanEncoder<int>();
        coeff_encoder.preprocess_encode(coeff_quant_inds, 0);
        coeff_encoder.save(compressed_data_pos);
        coeff_encoder.encode(coeff_quant_inds, compressed_data_pos);
        coeff_encoder.postprocess_encode();
    }
    auto encoder = SZ3::HuffmanEncoder<int>();
    encoder.preprocess_encode(quant_inds, 0);
    encoder.save(compressed_data_pos);
    encoder.encode(quant_inds, compressed_data_pos);
    encoder.postprocess_encode();
    auto lossless = SZ3::Lossless_zstd();
    std::cout << "size before lossless = " << compressed_data_pos - compressed << std::endl;
    compressed_size = 0;
    unsigned char *lossless_data = lossless.compress(compressed, compressed_data_pos - compressed, compressed_size);
    lossless.postcompress_data(compressed);
    std::cout << "compressed_size = " << compressed_size << std::endl;
    std::cout << "compression_ratio = " << num_elements*sizeof(T) * 1.0 / compressed_size << std::endl;
    std::cout << "Max value = " << max_val << ", min value = " << min_val << ", valid range = " << valid_range << std::endl;
    std::cout << "Max error = " << max_error << std::endl;
    double mse = total_error/num_elements;
    std::cout << "MSE = " << mse << std::endl;
    std::cout << "PSNR = " << 20 * log10((max_val - min_val) / sqrt(mse)) << std::endl;
    return lossless_data;
}

template <class T>
unsigned char * umc_compress_blockwise_adaptive2(const std::vector<int32_t>& index_map, int d, const int optimal_n_neighbors, const int optimal_reg_size, int blockwise_sample_size, const std::string& data_file, 
                                                    const std::vector<int32_t>& sampleNodes_index, const std::vector<uint32_t>& num_part_nodes, const std::vector<uint32_t>& part_map_inv, const std::vector<std::vector<AdjNode<T>>>& processed_adj_nodes,
                                                        const std::vector<gsl_matrix *>& design_matrix, std::vector<gsl_matrix *>& multiplier, const std::vector<T>& positions, size_t& compressed_size, double eb, bool use_abs_eb){
    struct timespec start, end, end1;
    int err = 0;
    double reorder_time = 0, compression_time = 0;
    err = clock_gettime(CLOCK_REALTIME, &start);
    size_t num_elements = 0;
    auto data = readfile<T>(data_file.c_str(), num_elements);
    assert(num_elements == processed_adj_nodes.size());
    const T invalid_val = -99999.0;
    if(index_map[0] != -1) data = reorder(data, index_map, num_elements, 1);  
    err = clock_gettime(CLOCK_REALTIME, &end);
    auto compressed = umc_compress_blockwise_adaptive2<T>(invalid_val, d, optimal_n_neighbors, optimal_reg_size, blockwise_sample_size, data, sampleNodes_index, num_part_nodes, part_map_inv, processed_adj_nodes, design_matrix, multiplier, positions, compressed_size, eb, use_abs_eb);
    err = clock_gettime(CLOCK_REALTIME, &end1);
    reorder_time = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
    printf("reorder time = %.6f\n", reorder_time);
    compression_time = (double)(end1.tv_sec - start.tv_sec) + (double)(end1.tv_nsec - start.tv_nsec)/(double)1000000000;
    printf("compression time = %.6f\n", compression_time);
    return compressed;
}

template <class T>
void umc_compress_blockwise_adaptive2(int reorder_opt, int d, const std::string& position_file, const std::string& conn_file, int from_edge, const std::vector<std::string>& data_files,
                                        int n_neighbors, int elements_per_partition, double blockwise_sampling_ratio, double eb, bool use_abs_eb){   
    struct timespec start, end;
    int err = 0;
    double _time = 0; 
    err = clock_gettime(CLOCK_REALTIME, &start);
    size_t num = 0;
    auto positions = readfile<T>(position_file.c_str(), num);
    size_t num_elements = num / d;
    auto conn = readfile<int>(conn_file.c_str(), num);
    auto processed_adj_list = generate_processed_adjacent_list(num_elements, d, conn, from_edge);
    std::vector<int32_t> index_map(num_elements, -1);
    auto adj_list = generate_adjacent_list(num_elements, d, conn, from_edge);
    if(reorder_opt > 0){
        index_map = generate_reorder_index_map(num_elements, adj_list, reorder_opt);
        positions = reorder(positions, index_map, num_elements, d);
        conn = reorder_conn(conn, index_map, d);
        processed_adj_list = generate_processed_adjacent_list(num_elements, d, conn, from_edge);
        adj_list = generate_adjacent_list(num_elements, d, conn, from_edge);
    }
    auto processed_adj_nodes = generate_adjacent_neighbors2(num_elements, d, processed_adj_list, positions);
    conn = std::vector<int>();
    // partition
    idx_t nparts = (num_elements - 1)/elements_per_partition + 1;
    auto part = (nparts == 1) ? std::vector<idx_t>(num_elements, 0) : create_partition(num_elements, nparts, adj_list);
    std::vector<uint32_t> part_map(num_elements, -1);
    std::vector<uint32_t> part_map_inv(num_elements, -1);
    std::vector<uint32_t> num_part_nodes(nparts, 0);
    for(int i=0; i<part.size(); i++){
        num_part_nodes[part[i]] ++;
    }
    init_part_map(part, num_part_nodes, part_map, part_map_inv);
    // node reordering
    auto part_map_new = blockDPFS(adj_list, part, part_map_inv, num_part_nodes);
    for(int i=0; i<num_elements; i++){
        part_map_inv[part_map_new[i]] = i;
    }
    // generate regression operator
    std::vector<gsl_matrix *> design_matrix(nparts);
    std::vector<gsl_matrix *> multiplier(nparts);
    uint32_t offset = 0;
    for(int i=0; i<nparts; i++){
        auto block_size = num_part_nodes[i];
        if(block_size > 0){
            std::vector<const T*> input_X;
            for(int j=0; j<block_size; j++){
                auto original_id = part_map_inv[offset+j];
                input_X.push_back(&positions[d*original_id]);
            }
            gsl_matrix *X = generate_design_matrix(block_size, d, input_X);
            design_matrix[i] = X;
            multiplier[i] = generate_multiplier2(num_part_nodes[i], d, X);
        }
        offset += block_size;
    }
    auto regenerated_adj_list = regenerate_processed_adjacent_list(num_elements, adj_list, part_map_new);
    processed_adj_nodes = generate_adjacent_neighbors(num_elements, d, n_neighbors, regenerated_adj_list, positions);
    int min_block_size = INT_MAX;
    for(auto& num:num_part_nodes){
        if(num > 0 && num < min_block_size){
            min_block_size = num;
        }
    }
    int blockwise_sample_size = (int)min_block_size * blockwise_sampling_ratio;
    std::cout << "min block size = " << min_block_size << ", " << "sample size = " << blockwise_sample_size << std::endl;
    std::mt19937 gen(46);
    std::uniform_int_distribution<int> dist(0, min_block_size - 1);
    std::vector<int32_t> sampleNodes_index(blockwise_sample_size);
    for(int k=0; k<blockwise_sample_size; k++){
        int rd_index = dist(gen);
        sampleNodes_index[k] = rd_index;
    }
    err = clock_gettime(CLOCK_REALTIME, &end);
    _time = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
    printf("preprocessing time = %.6f\n", _time);
    for(int i=0; i<data_files.size(); i++){
        size_t compressed_size = 0;
        auto compressed = umc_compress_blockwise_adaptive2<T>(index_map, d, n_neighbors, elements_per_partition, blockwise_sample_size, data_files[i], sampleNodes_index, num_part_nodes, part_map_inv, 
                                                                processed_adj_nodes, design_matrix, multiplier, positions, compressed_size, eb, use_abs_eb);
        writefile((data_files[i] + ".umc").c_str(), compressed, compressed_size);
    }
}


// Compression Method 7: SD
template <class T>
unsigned char * umc_compress_sd(const T invalid_val, size_t num_elements, const std::vector<T>& data, const std::vector<std::set<int32_t>>& adj_list, size_t& compressed_size, double eb, bool use_abs_eb){
    std::vector<T> refined_data(data);
    refined_data.erase(std::remove(refined_data.begin(), refined_data.end(), invalid_val), refined_data.end());
    auto max_val = *std::max_element(refined_data.begin(), refined_data.end());
    auto min_val = *std::min_element(refined_data.begin(), refined_data.end());
    auto valid_range = max_val - min_val;
    refined_data = std::vector<T>();
    if(!use_abs_eb){
        eb *= valid_range;
    }
    std::vector<std::pair<int32_t, T>> sorted_nodes;
    for (int i=0; i<num_elements; i++) {
        sorted_nodes.push_back(std::make_pair(i, data[i]));
    }
    // sort nodes by value in non-decreasing order
    std::sort(sorted_nodes.begin(), sorted_nodes.end(), sortByValue<T>);
    // SD step 1: set-based decomposition
    auto sbd1_sets = generate_sbd1_sets(num_elements, sorted_nodes, eb);
    size_t num_sbd1_sets = sbd1_sets.size();
    // SD step 2:  region-based decomposition of each sbd1_set
    std::vector<int> regionIds(num_elements, -1);
    init_rbd2_regions_map(sbd1_sets, adj_list, regionIds);
    size_t num_regions = *std::max_element(regionIds.begin(), regionIds.end()) + 1;
    // generate array Q (namely region_mean)
    auto region_mean = generate_region_mean(num_elements, num_regions, regionIds, data);
    // indentify boundary and interior vertices and generate array N_B
    std::vector<std::set<int32_t>> region_boundary(num_regions);
    std::vector<std::set<int32_t>> region_interior(num_regions);
    std::vector<char> visited(num_elements, false);
    auto N_B = split_and_generate_NB(num_elements, num_regions, adj_list, regionIds, region_boundary, region_interior, visited);
    // select seeds and generate array N_I
    std::vector<std::set<int32_t>> region_seeds(num_regions);
    auto N_I = select_seeds_and_generate_NI(num_regions, adj_list, visited, region_interior, region_seeds);
    // differential encoding and generate array I_B, I_I
    std::vector<int32_t> I_B, I_I;
    differential_encoder(num_regions, region_boundary, region_seeds, N_B, N_I, I_B, I_I);
    size_t num_boundary_nodes = I_B.size();
    size_t num_seeds = I_I.size();
    // statistics
    double max_error = 0;
    double squared_error = 0;
    for(int i=0; i<num_elements; i++){
        auto pred = region_mean[regionIds[i]];
        double error = data[i] - pred;
        if(fabs(error) > max_error) max_error = fabs(error);
        squared_error += error*error;
    }
    // lossless
    unsigned char * compressed = (unsigned char *) malloc(num_elements * 4 * sizeof(T));
    unsigned char * compressed_data_pos = compressed;
    write(eb, compressed_data_pos);
    write(num_regions, compressed_data_pos);
    write(num_boundary_nodes, compressed_data_pos);
    write(num_seeds, compressed_data_pos);
    write_vector(region_mean, compressed_data_pos);
    write_vector(N_B, compressed_data_pos);
    write_vector(N_I, compressed_data_pos);
    write_vector(I_B, compressed_data_pos);
    if(num_seeds > 0) write_vector(I_I, compressed_data_pos);
    auto lossless = SZ3::Lossless_zstd();
    std::cout << "size before lossless = " << compressed_data_pos - compressed << std::endl;
    unsigned char *lossless_data = lossless.compress(compressed, compressed_data_pos - compressed, compressed_size);
    lossless.postcompress_data(compressed);   
    std::cout << "compressed_size = " << compressed_size << std::endl;
    std::cout << "compression_ratio = " << num_elements*sizeof(T) * 1.0 / compressed_size << std::endl;
    std::cout << "Max value = " << max_val << ", min value = " << min_val << ", valid range = " << valid_range << std::endl;
    std::cout << "Max error = " << max_error << std::endl;
    double mse = squared_error/num_elements;
    std::cout << "MSE = " << mse << std::endl;
    std::cout << "PSNR = " << 20 * log10((max_val - min_val) / sqrt(mse)) << std::endl;
    return lossless_data;
}

template <class T>
unsigned char * umc_compress_sd(const std::vector<int32_t>& index_map, const std::string& data_file, const std::vector<std::set<int32_t>>& adj_list, size_t& compressed_size, double eb, bool use_abs_eb){
	struct timespec start, end, end1;
    int err = 0;
    double reorder_time = 0, compression_time = 0;;
    err = clock_gettime(CLOCK_REALTIME, &start);
    size_t num_elements = 0;
	auto data = readfile<T>(data_file.c_str(), num_elements);
    if(index_map[0] != -1) data = reorder(data, index_map, num_elements, 1);
    err = clock_gettime(CLOCK_REALTIME, &end);
    const T invalid_val = -99999.0;
    auto compressed = umc_compress_sd<T>(invalid_val, num_elements, data, adj_list, compressed_size, eb, use_abs_eb);
    err = clock_gettime(CLOCK_REALTIME, &end1);
    reorder_time = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
    printf("reorder time = %.6f\n", reorder_time);
    compression_time = (double)(end1.tv_sec - start.tv_sec) + (double)(end1.tv_nsec - start.tv_nsec)/(double)1000000000;
    printf("compression time = %.6f\n", compression_time);
    return compressed;
}

template <class T>
void umc_compress_sd(int reorder_opt, int d, const std::string& conn_file, int from_edge, const std::vector<std::string>& data_files, double eb, bool use_abs_eb){
    struct timespec start, end;
    int err = 0;
    double preprocessing_time = 0, _time = 0;
    err = clock_gettime(CLOCK_REALTIME, &start);
    size_t num = 0;
    auto conn = readfile<int>(conn_file.c_str(), num);
    auto n_data = *std::max_element(conn.begin(), conn.end()) + 1;
    size_t num_elements = static_cast<size_t>(n_data);
    printf("num_elements = %zu\n", num_elements);
    auto adj_list = generate_adjacent_list(num_elements, d, conn, from_edge);
    std::vector<int32_t> index_map(num_elements, -1);
    if(reorder_opt > 0){
        index_map = generate_reorder_index_map(num_elements, adj_list, reorder_opt);
        conn = reorder_conn(conn, index_map, d);
        adj_list = generate_adjacent_list(num_elements, d, conn, from_edge);
    }
    err = clock_gettime(CLOCK_REALTIME, &end);
    preprocessing_time = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
    printf("preprocessing time = %.6f\n", preprocessing_time);
    for(int i=0; i<data_files.size(); i++){
        size_t compressed_size = 0;
        auto compressed = umc_compress_sd<T>(index_map, data_files[i], adj_list, compressed_size, eb, use_abs_eb);
        writefile((data_files[i] + ".umc").c_str(), compressed, compressed_size);
    }
}

}
#endif