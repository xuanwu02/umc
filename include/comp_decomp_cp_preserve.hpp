#ifndef _UMC_COMP_DECOMP_CP_HPP
#define _UMC_COMP_DECOMP_CP_HPP

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <cmath>
#include <vector>
#include <time.h>
#include <unordered_set>
#include <SZ3/quantizer/IntegerQuantizer.hpp>
#include <SZ3/encoder/HuffmanEncoder.hpp>
#include <SZ3/lossless/Lossless_zstd.hpp>
#include "utils.hpp"
#include "reorder.hpp"
#include "adjacent_prediction.hpp"
#include "cp_preservation.hpp"

namespace UMC{

// Compression
template <class T>
unsigned char *
umc_compress_prediction_by_adjacent_nodes_cp_preserve(int n, const T * points, const T * data, int m, const std::vector<Tet>& tets, const std::vector<std::vector<std::pair<int, int>>>& point_tets, const std::vector<std::vector<AdjNode<T>>>& processed_adj_nodes, size_t& compressed_size, double max_pwr_eb){
    // map point to adjcent tets and position in that tets
    size_t sign_map_size = (3*n - 1)/8 + 1;
    unsigned char * sign_map = (unsigned char *) malloc(3*n*sizeof(unsigned char));
    T * log_data = log_transform(data, sign_map, 3*n);
    unsigned char * sign_map_compressed = (unsigned char *) malloc(sign_map_size);
    unsigned char * sign_map_compressed_pos = sign_map_compressed;
    convertIntArray2ByteArray_fast_1b_to_result_sz(sign_map, 3*n, sign_map_compressed_pos);
    free(sign_map);

    std::vector<T> eb_zero_data = std::vector<T>();
    double threshold = std::numeric_limits<float>::epsilon();
    const int base = 2;
    const double log_of_base = log2(base);
    const int capacity = 65536;
    const int intv_radius = (capacity >> 1);
    const int eb_quant_index_max = (int) (log2(1.0 / threshold)/log_of_base) + 1;
    
    T * dec_data = (T *) malloc(3*n*sizeof(T));
    memcpy(dec_data, data, 3*n*sizeof(T));
    T * dec_data_pos = dec_data;
    T * log_data_pos = log_data;
    int * eb_quant_index = (int *) malloc(n*sizeof(int));
    int * data_quant_index = (int *) malloc(3*n*sizeof(int));
    int * eb_quant_index_pos = eb_quant_index;
    int * data_quant_index_pos = data_quant_index;
    int count = 0;
    for(int i=0; i<n; i++){
        double required_eb = max_pwr_eb;
        auto adj_tets_ids = point_tets[i];
        for(const auto& id:adj_tets_ids){
            auto t = tets[id.first];
            int pos = id.second;
            std::vector<int> inds{0, 1, 2, 3};
            inds.erase(inds.begin() + pos);
            const int data_offset[4] = {t.vertex[inds[0]]*3, t.vertex[inds[1]]*3, t.vertex[inds[2]]*3, t.vertex[pos]*3};
            required_eb = MIN_(required_eb, max_eb_to_keep_position_and_type_3d_online(
                dec_data[data_offset[0]], dec_data[data_offset[1]], dec_data[data_offset[2]], dec_data[data_offset[3]],
                dec_data[data_offset[0] + 1], dec_data[data_offset[1] + 1], dec_data[data_offset[2] + 1], dec_data[data_offset[3] + 1],
                dec_data[data_offset[0] + 2], dec_data[data_offset[1] + 2], dec_data[data_offset[2] + 2], dec_data[data_offset[3] + 2]));
        }
        if(required_eb < 1e-10) required_eb = 0;
        if(required_eb > 0){
            bool unpred_flag = false;
            double abs_eb = log2(1 + required_eb);
            *eb_quant_index_pos = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);
            T decompressed[3];
            if(*eb_quant_index_pos > 0){
                // compress vector fields
                for(int p=0; p<3; p++){
                    T * cur_log_data_pos = log_data_pos + p;
                    T cur_data = *cur_log_data_pos;
                    T pred = 0;
                    for(int neighbor=0; neighbor<processed_adj_nodes[i].size(); neighbor++){
                        // neighbor index
                        int j = processed_adj_nodes[i][neighbor].id;
                        pred += processed_adj_nodes[i][neighbor].weight * log_data[j*3 + p];
                    }
                    double diff = cur_data - pred;
                    double quant_diff = fabs(diff) / abs_eb + 1;
                    if(quant_diff < capacity){
                        quant_diff = (diff > 0) ? quant_diff : -quant_diff;
                        int quant_index = (int)(quant_diff/2) + intv_radius;
                        data_quant_index_pos[p] = quant_index;
                        decompressed[p] = pred + 2 * (quant_index - intv_radius) * abs_eb; 
                        if(fabs(decompressed[p] - cur_data) >= abs_eb){
                            unpred_flag = true;
                            break;
                        }
                    }
                    else{
                        unpred_flag = true;
                        break;
                    }
                }
            }
            else unpred_flag = true;
            if(unpred_flag){
                *(eb_quant_index_pos ++) = 0;
                eb_zero_data.push_back(dec_data_pos[0]);
                eb_zero_data.push_back(dec_data_pos[1]);
                eb_zero_data.push_back(dec_data_pos[2]);
            }
            else{
                eb_quant_index_pos ++;
                data_quant_index_pos += 3;
                log_data_pos[0] = decompressed[0];
                log_data_pos[1] = decompressed[1];
                log_data_pos[2] = decompressed[2];
                dec_data_pos[0] = (dec_data_pos[0] > 0) ? exp2(log_data_pos[0]) : -exp2(log_data_pos[0]);
                dec_data_pos[1] = (dec_data_pos[1] > 0) ? exp2(log_data_pos[1]) : -exp2(log_data_pos[1]);
                dec_data_pos[2] = (dec_data_pos[2] > 0) ? exp2(log_data_pos[2]) : -exp2(log_data_pos[2]);
            }
        }
        else{
            count ++;
            *(eb_quant_index_pos ++) = 0;
            eb_zero_data.push_back(dec_data_pos[0]);
            eb_zero_data.push_back(dec_data_pos[1]);
            eb_zero_data.push_back(dec_data_pos[2]);
        }
        log_data_pos += 3;
        dec_data_pos += 3;
    }
    free(dec_data);
    free(log_data);
    unsigned char * compressed = (unsigned char *) malloc(3*n*sizeof(T));
    unsigned char * compressed_pos = compressed;
    write_variable_to_dst(compressed_pos, base);
    write_variable_to_dst(compressed_pos, intv_radius);
    write_array_to_dst(compressed_pos, sign_map_compressed, sign_map_size);
    free(sign_map_compressed);
    size_t unpredictable_count = eb_zero_data.size();
    write_variable_to_dst(compressed_pos, unpredictable_count);
    write_array_to_dst(compressed_pos, (T *)&eb_zero_data[0], unpredictable_count);
    size_t eb_quant_num = eb_quant_index_pos - eb_quant_index;
    write_variable_to_dst(compressed_pos, eb_quant_num);
    Huffman_encode_tree_and_data(2*256, eb_quant_index, n, compressed_pos);
    free(eb_quant_index);
    size_t data_quant_num = data_quant_index_pos - data_quant_index;
    write_variable_to_dst(compressed_pos, data_quant_num);
    Huffman_encode_tree_and_data(2*capacity, data_quant_index, data_quant_num, compressed_pos);
    free(data_quant_index);
    compressed_size = compressed_pos - compressed;
    return compressed;  
}

template <class T>
void umc_compress_prediction_by_adjacent_nodes_cp_preserve(int reorder_opt, const std::string& position_file, const std::string& conn_file, int from_edge, const std::string& Data_file, int d, double max_pwr_eb){
	struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);
    double setup_time = 0;	
    size_t num = 0;
    auto positions = readfile<T>(position_file.c_str(), num);
    assert(num % d == 0);
    size_t num_elements = num / d;
    printf("num_elements = %zu\n", num_elements);
    auto conn = readfile<int>(conn_file.c_str(), num);
    assert(num % (d+1) == 0);
    size_t num_cells = num / (d+1);
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
    auto processed_adj_nodes = generate_adjacent_neighbors(num_elements, d, processed_adj_list, positions);
    processed_adj_list = std::vector<std::set<int32_t>>();
    std::vector<std::vector<std::pair<int, int>>> point_tets;
    std::vector<Tet> tets = construct_tets(num_elements, num_cells, conn.data(), point_tets);
    conn = std::vector<int>();
    auto data_ori = readfile<T>(Data_file.c_str(), num);
    if(reorder_opt > 0) data_ori = reorder(data_ori, index_map, num_elements, d);
    clock_gettime(CLOCK_REALTIME, &end);
    setup_time = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
    printf("setup time = %.6f\n", setup_time);
    double compression_time = 0;
    clock_gettime(CLOCK_REALTIME, &start);
	size_t compressed_size = 0;
	auto compressed = umc_compress_prediction_by_adjacent_nodes_cp_preserve<T>(num_elements, positions.data(), data_ori.data(), num_cells, tets, point_tets, processed_adj_nodes, compressed_size, max_pwr_eb);
    point_tets = std::vector<std::vector<std::pair<int, int>>>();
    tets = std::vector<Tet>();
    auto lossless = SZ3::Lossless_zstd();
    size_t lossless_size = 0;
    unsigned char *lossless_data = lossless.compress(compressed, compressed_size, lossless_size);
    lossless.postcompress_data(compressed);
	writefile((Data_file + ".umc").c_str(), lossless_data, lossless_size);
	clock_gettime(CLOCK_REALTIME, &end);
    compression_time = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
    printf("compression time = %.6f\n", compression_time);
    printf("compression ratio = %.6f\n", 1.0 * num_elements * 3 *sizeof(T) / lossless_size);
    free(lossless_data);
    data_ori = std::vector<T>();
}


// Decompression
template <class T>
void
umc_decompress_prediction_by_adjacent_nodes_cp_preserve(const unsigned char * compressed, int num_elements, const std::vector<std::vector<AdjNode<T>>>& processed_adj_nodes, T *& data){
    if(data) free(data);
    auto n = num_elements;
    const unsigned char * compressed_pos = compressed;
    int base = 0;
    read_variable_from_src(compressed_pos, base);
    printf("base = %d\n", base);
    int intv_radius = 0;
    read_variable_from_src(compressed_pos, intv_radius);
    size_t sign_map_size = (3*n - 1)/8 + 1;
    unsigned char * sign_map = convertByteArray2IntArray_fast_1b_sz(3*n, compressed_pos, sign_map_size);    
    const int capacity = (intv_radius << 1);
    size_t unpred_data_count = 0;
    read_variable_from_src(compressed_pos, unpred_data_count);
    const T * eb_zero_data = (T *) compressed_pos;
    const T * eb_zero_data_pos = eb_zero_data;
    compressed_pos += unpred_data_count*sizeof(T);
    size_t eb_quant_num = 0;
    read_variable_from_src(compressed_pos, eb_quant_num);
    int * eb_quant_index = Huffman_decode_tree_and_data(2*256, eb_quant_num, compressed_pos);
    size_t data_quant_num = 0;
    read_variable_from_src(compressed_pos, data_quant_num);
    int * data_quant_index = Huffman_decode_tree_and_data(2*capacity, data_quant_num, compressed_pos);
    data = (T *) malloc(3*n*sizeof(T));
    T * data_pos = data;
    int * eb_quant_index_pos = eb_quant_index;
    int * data_quant_index_pos = data_quant_index;
    const double threshold=std::numeric_limits<float>::epsilon();
    double log_of_base = log2(base);
    int eb_quant_index_max = (int) (log2(1.0 / threshold)/log_of_base) + 1;
    std::unordered_set<int> unpred_data_indices;
    for(int i=0; i<n; i++){
        if(*eb_quant_index_pos == 0 || *eb_quant_index_pos == eb_quant_index_max){
            unpred_data_indices.insert(i);
            for(int p=0; p<3; p++){
                T cur_data = *(eb_zero_data_pos ++);
                data_pos[p] = (cur_data == 0) ? -100 : log2f(fabs(cur_data));
            }
            eb_quant_index_pos ++;
        }
        else{
            double eb = (*eb_quant_index_pos == 0) ? 0 : pow(base, *eb_quant_index_pos) * threshold;
            eb_quant_index_pos ++;
            for(int p=0; p<3; p++){
                T * cur_log_data_pos = data_pos + p;                    
                T pred = 0;
                for(int neighbor=0; neighbor<processed_adj_nodes[i].size(); neighbor++){
                    // neighbor index
                    int j = processed_adj_nodes[i][neighbor].id;
                    pred += processed_adj_nodes[i][neighbor].weight * data[j*3 + p];
                }
                *cur_log_data_pos = pred + 2 * (data_quant_index_pos[p] - intv_radius) * eb;
            }
            data_quant_index_pos += 3;
        }
        data_pos += 3;
    }
    printf("recover data done\n");
    eb_zero_data_pos = eb_zero_data;
    unsigned char * sign_pos = sign_map;
    for(int i=0; i<n; i++){
        if(unpred_data_indices.count(i)){
            for(int p=0; p<3; p++){
                data[3*i + p] = *(eb_zero_data_pos++);
            }
            sign_pos += 3;
        }
        else{
            for(int p=0; p<3; p++){
                if(data[3*i + p] < -99) data[3*i + p] = 0;
                else data[3*i + p] = *(sign_pos ++) ? exp2(data[3*i + p]) : -exp2(data[3*i + p]);
            }
        }
    }
    free(sign_map);
    free(eb_quant_index);
    free(data_quant_index);
}

template <class T>
void
umc_decompress_prediction_by_adjacent_nodes_cp_preserve(int reorder_opt, const std::string& position_file, const std::string& conn_file, int from_edge, const std::string& compressed_file, const std::vector<std::string>& data_files, int d){
	struct timespec start, end;
    int err = 0;
    err = clock_gettime(CLOCK_REALTIME, &start);
    double setup_time = 0;
    size_t num = 0;
    auto positions = readfile<T>(position_file.c_str(), num);
    assert(num % d == 0);
    size_t num_elements = num / d;
    auto conn = readfile<int>(conn_file.c_str(), num);
    assert(num % (d+1) == 0);
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
    auto processed_adj_nodes = generate_adjacent_neighbors(num_elements, d, processed_adj_list, positions);
    processed_adj_list = std::vector<std::set<int32_t>>();
    err = clock_gettime(CLOCK_REALTIME, &end);
    setup_time = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
    printf("setup time = %.6f\n", setup_time);
    T * dec_data = (T *) malloc(d*num_elements*sizeof(T));
    double compression_time = 0;
    err = clock_gettime(CLOCK_REALTIME, &start);
    size_t input_size = 0;
    auto input = readfile<unsigned char>(compressed_file.c_str(), input_size);
    auto lossless = SZ3::Lossless_zstd();
    auto compressed = lossless.decompress(input.data(), input_size);
    umc_decompress_prediction_by_adjacent_nodes_cp_preserve<T>(compressed, num_elements, processed_adj_nodes, dec_data);
	err = clock_gettime(CLOCK_REALTIME, &end);
    compression_time = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
    printf("decompression time = %.6f\n", compression_time);
    free(compressed);
    T * dec_data_t = (T *) malloc(num_elements*sizeof(T));
    for(int t=0; t<d; t++){
        memset(dec_data_t, 0, num_elements*sizeof(T));
        if(reorder_opt > 0){
            for(int i=0; i<num_elements; i++) dec_data_t[i] = dec_data[d*index_map[i] + t];
        }else{
            for(int i=0; i<num_elements; i++) dec_data_t[i] = dec_data[d*i + t];
        }
        auto data_ori = readfile<T>(data_files[t].c_str(), num);
        print_statistics(data_ori.data(), dec_data_t, num_elements);
        writefile((data_files[t] + ".out").c_str(), dec_data_t, num_elements);    
    }
    free(dec_data_t);
    free(dec_data);
}

}
#endif