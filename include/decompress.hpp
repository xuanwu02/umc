#ifndef _UMC_DECOMPRESS_HPP
#define _UMC_DECOMPRESS_HPP

#include <vector>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <cmath>
#include <time.h>
#include <SZ3/quantizer/IntegerQuantizer.hpp>
#include <SZ3/encoder/HuffmanEncoder.hpp>
#include <SZ3/lossless/Lossless_zstd.hpp>
#include "utils.hpp"
#include "reorder.hpp"
#include "partition.hpp"
#include "regression.hpp"
#include "adjacent_prediction.hpp"
#include "kriging_interpolation.hpp"
#include "set_decomposition.hpp" 
#include "cp_preservation.hpp" 

namespace UMC{


// Decompression Method 2: RBP
template <class T>
void umc_decompress_prediction_by_regression(int d, const std::vector<int32_t>& index_map, int elements_per_partition, const std::string& compressed_file, const std::string& data_file, 
                                                const std::vector<gsl_matrix *>& design_matrix, const std::vector<uint32_t>& num_part_nodes, const std::vector<uint32_t>& part_map_inv, T *& data){
    size_t num_elements = 0;
    std::vector<T> data_ori = readfile<T>(data_file.c_str(), num_elements);
    if(index_map[0] != -1) data_ori = reorder(data_ori, index_map, num_elements, 1);
    size_t input_size = 0;
    auto input = readfile<unsigned char>(compressed_file.c_str(), input_size);
    auto lossless = SZ3::Lossless_zstd();
    size_t remaining_length = input_size;
    auto compressed_data = lossless.decompress(input.data(), input_size);
    const unsigned char *compressed_data_pos = compressed_data;
    double eb = 0;
    read(eb, compressed_data_pos);
    size_t num_coeff_quant_inds = 0;
    read(num_coeff_quant_inds, compressed_data_pos);
    size_t num_quant_inds = 0;
    read(num_quant_inds, compressed_data_pos);
    const int quant_radius = 32768;
    const double reg_coeff_eb = 0.1;
    SZ3::LinearQuantizer<T> quantizer = SZ3::LinearQuantizer<T>(eb, quant_radius);;
    SZ3::LinearQuantizer<T> coeff_quantizer = SZ3::LinearQuantizer<T>(reg_coeff_eb / (d+1), quant_radius);
    std::vector<int> quant_inds;
    std::vector<int> coeff_quant_inds;
    coeff_quantizer.load(compressed_data_pos, remaining_length);
    quantizer.load(compressed_data_pos, remaining_length);
    auto coeff_encoder = SZ3::HuffmanEncoder<int>();
    coeff_encoder.load(compressed_data_pos, remaining_length);
    coeff_quant_inds = coeff_encoder.decode(compressed_data_pos, num_coeff_quant_inds);
    coeff_encoder.postprocess_decode();
    auto encoder = SZ3::HuffmanEncoder<int>();
    encoder.load(compressed_data_pos, remaining_length);
    quant_inds = encoder.decode(compressed_data_pos, num_quant_inds);
    encoder.postprocess_decode();
    lossless.postdecompress_data(compressed_data);
    const int * quant_inds_pos = quant_inds.data();
	const int * coeff_quant_inds_pos = coeff_quant_inds.data();
    std::vector<T> reg_coeff(d+1, 0);
    std::vector<T> output_Y(elements_per_partition * 2, 0);
    uint32_t offset = 0;
    for(int i=0; i<num_part_nodes.size(); i++){
        if(num_part_nodes[i] == 0) continue;
        est_and_recover(num_part_nodes[i], d, design_matrix[i], quantizer, quant_inds_pos, coeff_quantizer, coeff_quant_inds_pos, reg_coeff, output_Y);
        auto * output_Y_pos = output_Y.data();
        for(int j=0; j<num_part_nodes[i]; j++){
            auto original_id = part_map_inv[j + offset];
            data[original_id] = *(output_Y_pos++);
        }
        offset += num_part_nodes[i];
    }
    const T invalid_val = -99999.0;
    print_statistics(data_ori.data(), data, num_elements, invalid_val);
}	

template <class T>
void umc_decompress_prediction_by_regression(int reorder_opt, int d, int elements_per_partition, const std::string& position_file, const std::string& conn_file, 
                                                int from_edge, const std::vector<std::string>& compressed_files, const std::vector<std::string>& data_files){	
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
    std::cout << "nparts = " << nparts << std::endl;
	std::vector<uint32_t> part_map(num_elements, -1);
	std::vector<uint32_t> part_map_inv(num_elements, -1);
	std::vector<uint32_t> num_part_nodes(nparts, 0);
	for(int i=0; i<part.size(); i++){
	    num_part_nodes[part[i]] ++;
	}
	init_part_map(part, num_part_nodes, part_map, part_map_inv);
	std::vector<gsl_matrix *> design_matrix(nparts);
	uint32_t offset = 0;
	for(int i=0; i<nparts; i++){
	    if(num_part_nodes[i] == 0) continue;
        std::vector<const T*> input_X;
        for(int j=0; j<num_part_nodes[i]; j++){
            auto original_id = part_map_inv[j + offset];
            input_X.push_back(&positions[d*original_id]);
        }
        gsl_matrix *X = generate_design_matrix(num_part_nodes[i], d, input_X);
        design_matrix[i] = X;
	    offset += num_part_nodes[i];
	}
    err = clock_gettime(CLOCK_REALTIME, &end);
    preprocessing_time = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
    printf("preprocessing time = %.6f\n", preprocessing_time);
    for(int i=0; i<data_files.size(); i++){
        double decompression_time = 0, reorder_time = 0;
        err = clock_gettime(CLOCK_REALTIME, &start);
        T * dec_data = (T *) malloc(num_elements*sizeof(T));
        umc_decompress_prediction_by_regression<T>(d, index_map, elements_per_partition, compressed_files[i], data_files[i], design_matrix, num_part_nodes, part_map_inv, dec_data);
		err = clock_gettime(CLOCK_REALTIME, &end);
        decompression_time = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
        printf("decompression time = %.6f\n", decompression_time);
        err = clock_gettime(CLOCK_REALTIME, &start);
        std::vector<T> data_recovered(num_elements, 0);
        if(reorder_opt > 0){
            for(int i=0; i<num_elements; i++){
                data_recovered[i] = dec_data[index_map[i]];
            }
        }
        err = clock_gettime(CLOCK_REALTIME, &end);
        reorder_time = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
        printf("reorder time = %.6f\n", reorder_time);
        writefile((compressed_files[i] + ".out").c_str(), data_recovered.data(), num_elements);
        free(dec_data);
    }
    for(int i=0; i<design_matrix.size(); i++){
    	gsl_matrix_free(design_matrix[i]);
    }
}


// Decompression Method 2: NBP-IDW
template <class T>
void umc_decompress_prediction_by_adjacent_nodes_simple(const std::vector<int32_t>& index_map, const std::string& compressed_file, const std::string& data_file, const std::vector<std::vector<AdjNode<T>>>& processed_adj_nodes, T *& data){
    size_t num_elements = 0;
    std::vector<T> data_ori = readfile<T>(data_file.c_str(), num_elements);
    if(index_map[0] != -1) data_ori = reorder(data_ori, index_map, num_elements, 1);
    size_t input_size = 0;
    auto input = readfile<unsigned char>(compressed_file.c_str(), input_size);
    auto lossless = SZ3::Lossless_zstd();
    size_t remaining_length = input_size;
    auto compressed_data = lossless.decompress(input.data(), input_size);
    const unsigned char *compressed_data_pos = compressed_data;
    int n_neighbors = 0;
    read(n_neighbors, compressed_data_pos);
    double eb = 0;
    read(eb, compressed_data_pos);
    size_t num_quant_inds = 0;
    read(num_quant_inds, compressed_data_pos);
    const int quant_radius = 32768;
    SZ3::LinearQuantizer<T> quantizer = SZ3::LinearQuantizer<T>(eb, quant_radius);
    std::vector<int> quant_inds;
    quantizer.load(compressed_data_pos, remaining_length);
    auto encoder = SZ3::HuffmanEncoder<int>();
    encoder.load(compressed_data_pos, remaining_length);
    quant_inds = encoder.decode(compressed_data_pos, num_quant_inds);
    encoder.postprocess_decode();
    lossless.postdecompress_data(compressed_data);
    const int * quant_inds_pos = quant_inds.data();
    for(int i=0; i<num_elements; i++){
    	T pred = 0;
    	for(int neighbor=0; neighbor<processed_adj_nodes[i].size(); neighbor++){
    		int j = processed_adj_nodes[i][neighbor].id;
    		pred += processed_adj_nodes[i][neighbor].weight * data[j];
    	}
        data[i] = quantizer.recover(pred, *(quant_inds_pos++));
    }
    const T invalid_val = -99999.0;
    print_statistics(data_ori.data(), data, num_elements, invalid_val);
}

template <class T>
void umc_decompress_prediction_by_adjacent_nodes_simple(int reorder_opt, int d, const std::string& position_file, const std::string& conn_file, int from_edge,
                                                            const std::vector<std::string>& compressed_files, const std::vector<std::string>& data_files){
	struct timespec start, end;
    int err = 0;
    err = clock_gettime(CLOCK_REALTIME, &start);
    double preprocessing_time = 0;
    // read parameters
    size_t input_size = 0;
    auto input = readfile<unsigned char>(compressed_files[0].c_str(), input_size);
    auto lossless = SZ3::Lossless_zstd();
    auto compressed_data = lossless.decompress(input.data(), input_size);
    const unsigned char *compressed_data_pos = compressed_data;
    int n_neighbors = 0;
    read(n_neighbors, compressed_data_pos);
    lossless.postdecompress_data(compressed_data);
    input = std::vector<unsigned char>();
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
        double decompression_time = 0, reorder_time = 0;
        err = clock_gettime(CLOCK_REALTIME, &start);
        T * dec_data = (T *) malloc(num_elements*sizeof(T));
        umc_decompress_prediction_by_adjacent_nodes_simple<T>(index_map, compressed_files[i], data_files[i], processed_adj_nodes, dec_data);
		err = clock_gettime(CLOCK_REALTIME, &end);
        decompression_time = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
        printf("decompression time = %.6f\n", decompression_time);
        err = clock_gettime(CLOCK_REALTIME, &start);
        std::vector<T> data_recovered(num_elements, 0);
        if(reorder_opt > 0){
            for(int i=0; i<num_elements; i++){
                data_recovered[i] = dec_data[index_map[i]];
            }
        }
        err = clock_gettime(CLOCK_REALTIME, &end);
        reorder_time = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
        printf("reorder time = %.6f\n", reorder_time);
        writefile((compressed_files[i] + ".out").c_str(), data_recovered.data(), num_elements);
        free(dec_data);
    }
}


// Decompression Method 3: NBP-KW
template <class T>
void umc_decompress_prediction_by_adjacent_nodes_kriging(int timestep, int batch, const std::vector<int32_t>& index_map, const std::string& compressed_file, const std::string& data_file, const std::vector<T>& positions,
                                                            std::vector<std::vector<AdjNode<T>>>& processed_adj_nodes, int d, T *& data){
    size_t num_elements = 0;
    std::vector<T> data_ori = readfile<T>(data_file.c_str(), num_elements);
    if(index_map[0] != -1) data_ori = reorder(data_ori, index_map, num_elements, 1);
    size_t input_size = 0;
    auto input = readfile<unsigned char>(compressed_file.c_str(), input_size);
    auto lossless = SZ3::Lossless_zstd();
    size_t remaining_length = input_size;
    auto compressed_data = lossless.decompress(input.data(), input_size);
    const unsigned char *compressed_data_pos = compressed_data;
    int min_neighbors = 0;
    read(min_neighbors, compressed_data_pos);
    int max_iter = 0;
    read(max_iter, compressed_data_pos);
    double eb = 0;
    read(eb, compressed_data_pos);
    size_t num_quant_inds = 0;
    read(num_quant_inds, compressed_data_pos);
    const int quant_radius = 32768;
    SZ3::LinearQuantizer<T> quantizer = SZ3::LinearQuantizer<T>(eb, quant_radius);
    std::vector<int> quant_inds;
    quantizer.load(compressed_data_pos, remaining_length);
    auto encoder = SZ3::HuffmanEncoder<int>();
    encoder.load(compressed_data_pos, remaining_length);
    quant_inds = encoder.decode(compressed_data_pos, num_quant_inds);
    encoder.postprocess_decode();
    lossless.postdecompress_data(compressed_data);
    const int * quant_inds_pos = quant_inds.data();
    std::cout << "absolute eb = " << eb << std::endl;
    bool renew = (timestep % batch == 0);
    for(int i=0; i<num_elements; i++){
    	T pred = 0;
        double denominator = 0;
        if(processed_adj_nodes[i].size() < min_neighbors){
            for(int neighbor=0; neighbor<processed_adj_nodes[i].size(); neighbor++){
                int j = processed_adj_nodes[i][neighbor].id;
                pred += processed_adj_nodes[i][neighbor].weight * data[j];
                denominator += processed_adj_nodes[i][neighbor].weight;
            }
            if(denominator > 0) pred /= denominator;
        }else{
            if(renew){
                std::vector<double> spatial_lag;
                std::vector<double> observed_semivariogram;
                for(int j=0; j<processed_adj_nodes[i].size(); j++){
                    int id_j = processed_adj_nodes[i][j].id;
                    for(int k=0; k<processed_adj_nodes[i].size(); k++){
                        if(k > j){
                            int id_k = processed_adj_nodes[i][k].id;
                            double diff_jk = data[id_j] - data[id_k];
                            observed_semivariogram.push_back(diff_jk * diff_jk / 2.0);
                            spatial_lag.push_back(distance_euclid(d, &positions[id_j*d], &positions[id_k*d]));
                        }
                    }
                }
                double scale_est = 4.0;
                double range_est = *std::max_element(spatial_lag.begin(), spatial_lag.end()) / sqrt(3);
                fit_model(spatial_lag, observed_semivariogram, range_est, scale_est, max_iter);
                update_kriging_weights(range_est, scale_est, i, d, processed_adj_nodes, positions);
            }
            for(int j=0; j<processed_adj_nodes[i].size(); j++){
                pred += processed_adj_nodes[i][j].krg_weight * data[processed_adj_nodes[i][j].id];
            }
        }
        data[i] = quantizer.recover(pred, *(quant_inds_pos++));
    }
    const T invalid_val = -99999.0;
    print_statistics(data_ori.data(), data, num_elements, invalid_val);
}

template <class T>
void umc_decompress_prediction_by_adjacent_nodes_kriging(int reorder_opt, int d, const std::string& position_file, const std::string& conn_file, int from_edge, int batch,
                                                            const std::vector<std::string>& compressed_files, const std::vector<std::string>& data_files){
	struct timespec start, end;
    int err = 0;
    err = clock_gettime(CLOCK_REALTIME, &start);
    double preprocessing_time = 0;
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
    auto processed_adj_nodes = generate_adjacent_neighbors2(num_elements, d, 4, processed_adj_list, positions);
    err = clock_gettime(CLOCK_REALTIME, &end);
    preprocessing_time = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
    printf("preprocessing time = %.6f\n", preprocessing_time);
    for(int i=0; i<data_files.size(); i++){
        double decompression_time = 0;
        err = clock_gettime(CLOCK_REALTIME, &start);
        T * dec_data = (T *) malloc(num_elements*sizeof(T));
        umc_decompress_prediction_by_adjacent_nodes_kriging<T>(i, batch, index_map, compressed_files[i], data_files[i], positions, processed_adj_nodes, d, dec_data);
		err = clock_gettime(CLOCK_REALTIME, &end);
        decompression_time = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
        printf("decompression time = %.6f\n", decompression_time);
        double reorder_time = 0;
        err = clock_gettime(CLOCK_REALTIME, &start);
        std::vector<T> data_recovered(num_elements, 0);
        if(reorder_opt > 0){
            for(int i=0; i<num_elements; i++){
                data_recovered[i] = dec_data[index_map[i]];
            }
        }
        err = clock_gettime(CLOCK_REALTIME, &end);
        reorder_time = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
        printf("reorder time = %.6f\n", reorder_time);
        writefile((compressed_files[i] + ".out").c_str(), data_recovered.data(), num_elements);
        free(dec_data);
    }
}


// Decompression Method 4: NBP-IDW-INP
template <class T>
void umc_decompress_prediction_by_adjacent_nodes_dynamic(const std::vector<std::set<int32_t>>& original_adj_list, int timestep, int batch, const std::vector<int32_t>& index_map, const std::string& compressed_file, const std::string& data_file,
                                                            unsigned char * is_invalid, const std::vector<std::vector<AdjNode<T>>>& ref_processed_adj_nodes,
                                                                std::vector<std::vector<AdjNode<T>>>& current_processed_adj_nodes, T *& data){
    size_t num_elements = 0;
    std::vector<T> data_ori = readfile<T>(data_file.c_str(), num_elements);
    if(index_map[0] != -1) data_ori = reorder(data_ori, index_map, num_elements, 1);
    size_t input_size = 0;
    auto input = readfile<unsigned char>(compressed_file.c_str(), input_size);
    auto lossless = SZ3::Lossless_zstd();
    size_t remaining_length = input_size;
    auto compressed_data = lossless.decompress(input.data(), input_size);
    const unsigned char *compressed_data_pos = compressed_data;
    T invalid_val = 0;
    read(invalid_val, compressed_data_pos);
    int n_neighbors = 0;
    read(n_neighbors, compressed_data_pos);
    double eb = 0;
    read(eb, compressed_data_pos);
    std::cout << "absolute eb = " << eb << std::endl;
    size_t num_quant_inds = 0;
    read(num_quant_inds, compressed_data_pos);
    const int quant_radius = 32768;
    SZ3::LinearQuantizer<T> quantizer = SZ3::LinearQuantizer<T>(eb, quant_radius);
    std::vector<int> quant_inds;
    quantizer.load(compressed_data_pos, remaining_length);
    unsigned char * changed_invalid = (unsigned char *)malloc(num_elements * sizeof(unsigned char));
    bool renew = (timestep % batch == 0);
    if(renew) changed_invalid = convertByteArray2IntArray_fast_1b_sz(num_elements, compressed_data_pos, (num_elements - 1)/8 + 1);
    auto encoder = SZ3::HuffmanEncoder<int>();
    encoder.load(compressed_data_pos, remaining_length);
    quant_inds = encoder.decode(compressed_data_pos, num_quant_inds);
    encoder.postprocess_decode();
    lossless.postdecompress_data(compressed_data);
    const int * quant_inds_pos = quant_inds.data();
    if(renew){
        for(int i=0; i<num_elements; i++){
            if(changed_invalid[i]) is_invalid[i] = !is_invalid[i];
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
                pred += wt * data[j];
                denominator += wt;
            }
            if(denominator > 0) pred /= denominator;
        }
        data[i] = quantizer.recover(pred, *(quant_inds_pos++));
    }
    print_statistics(data_ori.data(), data, num_elements, invalid_val);
    free(changed_invalid);
}

template <class T>
void umc_decompress_prediction_by_adjacent_nodes_dynamic(int reorder_opt, int d, const std::string& position_file, const std::string& conn_file, int from_edge, int batch,
                                                            const std::vector<std::string>& compressed_files, const std::vector<std::string>& data_files){
	struct timespec start, end;
    int err = 0;
    err = clock_gettime(CLOCK_REALTIME, &start);
    double preprocessing_time = 0;
    size_t num = 0;
    auto positions = readfile<T>(position_file.c_str(), num);
    assert(num % d == 0);
    size_t num_elements = num / d;
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
        double decompression_time = 0;
        err = clock_gettime(CLOCK_REALTIME, &start);
        T * dec_data = (T *) malloc(num_elements*sizeof(T));
        umc_decompress_prediction_by_adjacent_nodes_dynamic<T>(original_adj_list, i, batch, index_map, compressed_files[i], data_files[i], is_invalid, ref_processed_adj_nodes, current_processed_adj_nodes, dec_data);
		err = clock_gettime(CLOCK_REALTIME, &end);
        decompression_time = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
        printf("decompression time = %.6f\n", decompression_time);
        double reorder_time = 0;
        err = clock_gettime(CLOCK_REALTIME, &start);
        std::vector<T> data_recovered(num_elements, 0);
        if(reorder_opt > 0){
            for(int i=0; i<num_elements; i++){
                data_recovered[i] = dec_data[index_map[i]];
            }
        }
        err = clock_gettime(CLOCK_REALTIME, &end);
        reorder_time = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
        printf("reorder time = %.6f\n", reorder_time);      
        writefile((compressed_files[i] + ".out").c_str(), data_recovered.data(), num_elements);
        free(dec_data);
    }
    free(is_invalid);
}


// Decompression Method 6: ADP-2
template <class T>
void umc_decompress_blockwise_adaptive2(const std::vector<int32_t> index_map, idx_t nparts, int d, const std::string& data_file,
                                const std::string& compressed_file, const std::vector<uint32_t>& num_part_nodes, const std::vector<uint32_t>& part_map_inv,
                                const std::vector<std::vector<AdjNode<T>>>& processed_adj_nodes, const std::vector<T>& positions, T *& data){
    size_t num_elements = 0;
    auto data_ori = readfile<T>(data_file.c_str(), num_elements);
    if(index_map[0] != -1) data_ori = reorder(data_ori, index_map, num_elements, 1);
    size_t input_size = 0;
    auto input = readfile<unsigned char>(compressed_file.c_str(), input_size);
    auto lossless = SZ3::Lossless_zstd();
    size_t remaining_length = input_size;
    auto compressed_data = lossless.decompress(input.data(), input_size);
    const unsigned char *compressed_data_pos = compressed_data;
    int n_neighbors = 0;
    read(n_neighbors, compressed_data_pos);
    int elements_per_partition = 0;
    read(elements_per_partition, compressed_data_pos);
    double eb = 0;
    read(eb, compressed_data_pos);
    // std::cout << "absolute eb = " << eb << std::endl;
    size_t num_coeff_quant_inds = 0;
    read(num_coeff_quant_inds, compressed_data_pos);
    size_t num_quant_inds = 0;
    read(num_quant_inds, compressed_data_pos);
    const int quant_radius = 32768;
    const double reg_coeff_eb = 0.1;
    SZ3::LinearQuantizer<T> quantizer = SZ3::LinearQuantizer<T>(eb, quant_radius);;
    SZ3::LinearQuantizer<T> coeff_quantizer = SZ3::LinearQuantizer<T>(reg_coeff_eb / (d+1), quant_radius);
    std::vector<int> quant_inds;
    std::vector<int> coeff_quant_inds;
    coeff_quantizer.load(compressed_data_pos, remaining_length);
    quantizer.load(compressed_data_pos, remaining_length);
    auto pred_method = convertByteArray2IntArray_fast_1b_sz(nparts, compressed_data_pos, (nparts - 1)/8 + 1);
    if(num_coeff_quant_inds > 0){
        auto coeff_encoder = SZ3::HuffmanEncoder<int>();
        coeff_encoder.load(compressed_data_pos, remaining_length);
        coeff_quant_inds = coeff_encoder.decode(compressed_data_pos, num_coeff_quant_inds);
        coeff_encoder.postprocess_decode();
    }
    auto encoder = SZ3::HuffmanEncoder<int>();
    encoder.load(compressed_data_pos, remaining_length);
    quant_inds = encoder.decode(compressed_data_pos, num_quant_inds);
    encoder.postprocess_decode();
    lossless.postdecompress_data(compressed_data);
    const int * quant_inds_pos = quant_inds.data();
	const int * coeff_quant_inds_pos = coeff_quant_inds.data();
    std::vector<T> reg_coeff(d+1, 0);
    uint32_t offset = 0;
    for(int i=0; i<nparts; i++){
        auto block_size = num_part_nodes[i];
        if(block_size == 0) continue;
        if(pred_method[i] == 0){
            for(int j=0; j<block_size; j++){
                auto original_id = part_map_inv[offset+j];
                T pred = 0;
                for(int neighbor=0; neighbor<processed_adj_nodes[original_id].size(); neighbor++){
                    int j = processed_adj_nodes[original_id][neighbor].id;
                    pred += processed_adj_nodes[original_id][neighbor].weight * data[j];
    	        }
                data[original_id] = quantizer.recover(pred, *(quant_inds_pos++));
            }
        }else{
            gsl_vector *c = gsl_vector_alloc(d+1);
            for(int k=0; k<d+1; k++){
                T dec_ck = coeff_quantizer.recover(reg_coeff[k], *(coeff_quant_inds_pos++));
                reg_coeff[k] = dec_ck;
                gsl_vector_set(c, k, dec_ck);
            }
            std::vector<const T*> input_X;
            for(int j=0; j<block_size; j++){
                auto original_id = part_map_inv[offset+j];
                input_X.push_back(&positions[d*original_id]);
            }
            gsl_vector *y_pred = gsl_vector_alloc(block_size);
            gsl_matrix *design_matrix = generate_design_matrix(block_size, d, input_X);
            gsl_blas_dgemv(CblasNoTrans, 1.0, design_matrix, c, 0, y_pred);
            for(int j=0; j<block_size; j++){
                auto original_id = part_map_inv[offset+j];
                T pred = gsl_vector_get(y_pred, j);
                data[original_id] = quantizer.recover(pred, *(quant_inds_pos++));
            }
        }
        offset += block_size;
    }
    free(pred_method);
    const T invalid_val = -99999.0;
    print_statistics(data_ori.data(), data, num_elements, invalid_val);
}

template <class T>
void umc_decompress_blockwise_adaptive2(int reorder_opt, int d, const std::string& position_file, const std::string& conn_file, int from_edge, const std::vector<std::string>& compressed_files, const std::vector<std::string>& data_files){
    struct timespec start, end;
    int err = 0;
    double _time = 0; 
    err = clock_gettime(CLOCK_REALTIME, &start);
    size_t num = 0;
    auto positions = readfile<T>(position_file.c_str(), num);
    assert(num % d == 0);
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
    // read parameters
    size_t input_size = 0;
    auto input = readfile<unsigned char>(compressed_files[0].c_str(), input_size);
    auto lossless = SZ3::Lossless_zstd();
    auto compressed_data = lossless.decompress(input.data(), input_size);
    const unsigned char *compressed_data_pos = compressed_data;
    int n_neighbors = 0;
    read(n_neighbors, compressed_data_pos);
    int reg_size = 0;
    read(reg_size, compressed_data_pos);
    double eb = 0;
    read(eb, compressed_data_pos);
    lossless.postdecompress_data(compressed_data);
    input = std::vector<unsigned char>();
    // partition
    idx_t nparts = (num_elements - 1)/reg_size + 1;
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
    auto regenerated_adj_list = regenerate_processed_adjacent_list(num_elements, adj_list, part_map_new);
    auto processed_adj_nodes = generate_adjacent_neighbors(num_elements, d, n_neighbors, regenerated_adj_list, positions);
    err = clock_gettime(CLOCK_REALTIME, &end);
    _time = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
    printf("preprocessing time = %.6f\n", _time);
    for(int i=0; i<data_files.size(); i++){
        err = clock_gettime(CLOCK_REALTIME, &start);
        T * dec_data = (T *) malloc(num_elements*sizeof(T));
        umc_decompress_blockwise_adaptive2<T>(index_map, nparts, d, data_files[i], compressed_files[i],
                                    num_part_nodes, part_map_inv, processed_adj_nodes, positions, dec_data);
        err = clock_gettime(CLOCK_REALTIME, &end);
        _time = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
        printf("decompression time = %.6f\n", _time);        
        err = clock_gettime(CLOCK_REALTIME, &start);
        std::vector<T> data_recovered(num_elements, 0);
        if(reorder_opt > 0){
            for(int i=0; i<num_elements; i++){
                data_recovered[i] = dec_data[index_map[i]];
            }
        }
        err = clock_gettime(CLOCK_REALTIME, &end);
        double reorder_time = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
        printf("reorder time = %.6f\n", reorder_time);
        writefile((compressed_files[i] + ".out").c_str(), data_recovered.data(), num_elements);
        free(dec_data);
    }
}


// Decompression Method 7: SD
template <class T>
void umc_decompress_sd(const std::vector<int32_t> index_map, const std::vector<std::set<int32_t>>& adj_list, const std::string& data_file, const std::string& compressed_file, T *& data){
    size_t num_elements = 0;
    std::vector<T> data_ori = readfile<T>(data_file.c_str(), num_elements);
    if(index_map[0] != -1) data_ori = reorder(data_ori, index_map, num_elements, 1);
    // read parameters
    size_t input_size = 0;
    auto input = readfile<unsigned char>(compressed_file.c_str(), input_size);
    auto lossless = SZ3::Lossless_zstd();
    auto compressed_data = lossless.decompress(input.data(), input_size);
    const unsigned char *compressed_data_pos = compressed_data;
    double eb;
    size_t num_regions, num_boundary_nodes, num_seeds;
    read(eb, compressed_data_pos);
    read(num_regions, compressed_data_pos);
    read(num_boundary_nodes, compressed_data_pos);
    read(num_seeds, compressed_data_pos);
    std::vector<T> region_mean(num_regions);
    std::vector<int> N_B(num_regions), N_I(num_regions), I_B(num_boundary_nodes), I_I(num_seeds);
    read_vector(region_mean, compressed_data_pos);
    read_vector(N_B, compressed_data_pos);
    read_vector(N_I, compressed_data_pos);
    read_vector(I_B, compressed_data_pos);
    if(num_seeds > 0) read_vector(I_I, compressed_data_pos);
    // reconstruction
    std::vector<T> dec_data(num_elements);
    std::vector<char> visited(num_elements, false);
    std::vector<int> boundary_index(num_regions);
    std::vector<int> seeds_index(num_regions, -1);
    // compute index offset of boundary and seeds
    int h = 0, offset = 0;
    while(h < num_regions){
        boundary_index[h] = offset;
        offset += N_B[h++];
    }
    h = 0; offset = 0;
    while(h < num_regions){ 
        if(N_I[h]) seeds_index[h] = offset;
        offset += N_I[h++];
    }
    recover_boundary_and_seeds(dec_data, visited, boundary_index, seeds_index, region_mean, N_B, I_B, N_I, I_I);
    recover_residual(dec_data, visited, adj_list, I_I);
    memcpy(data, dec_data.data(), num_elements*sizeof(T));
    const T invalid_val = -99999.0;
    print_statistics(data_ori.data(), data, num_elements, invalid_val);
}

template <class T>
void umc_decompress_sd(int reorder_opt, int d, const std::string& conn_file, int from_edge, const std::vector<std::string>& compressed_files, const std::vector<std::string>& data_files){
	struct timespec start, end;
    int err = 0;
    err = clock_gettime(CLOCK_REALTIME, &start);
    double preprocessing_time = 0;
    size_t num = 0;
    auto conn = readfile<int>(conn_file.c_str(), num);
    auto n_data = *std::max_element(conn.begin(), conn.end()) + 1;
    size_t num_elements = static_cast<size_t>(n_data);  
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
        double decompression_time = 0, reorder_time = 0;
        err = clock_gettime(CLOCK_REALTIME, &start);
        T * dec_data = (T *) malloc(num_elements*sizeof(T));
        umc_decompress_sd<T>(index_map, adj_list, data_files[i], compressed_files[i], dec_data);
		err = clock_gettime(CLOCK_REALTIME, &end);
        decompression_time = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
        printf("decompression time = %.6f\n", decompression_time);
        err = clock_gettime(CLOCK_REALTIME, &start);
        std::vector<T> data_recovered(num_elements, 0);
        memcpy(data_recovered.data(), dec_data, num_elements*sizeof(T));
        if(reorder_opt > 0){
            for(int i=0; i<num_elements; i++){
                data_recovered[i] = dec_data[index_map[i]];
            }
        }
        err = clock_gettime(CLOCK_REALTIME, &end);
        reorder_time = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
        printf("reorder time = %.6f\n", reorder_time);
        writefile((compressed_files[i] + ".out").c_str(), data_recovered.data(), num_elements);
        free(dec_data);
    }
}

}
#endif