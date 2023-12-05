#include <iostream>
#include <stdlib.h>
#include <string>
#include <stdio.h>
#include <vector>
#include <cmath>
#include "compress.hpp"

using namespace UMC;

int main(int argc, char ** argv){

    using T = float;
    int arg_pos = 1;
    std::string position_file(argv[arg_pos++]);
    std::string conn_file(argv[arg_pos++]);
    int from_edge = atoi(argv[arg_pos++]);
    int d = atoi(argv[arg_pos++]);
    int comp_opt = atoi(argv[arg_pos++]);
    int reorder_opt = atoi(argv[arg_pos++]);
    int num_data_files = atoi(argv[arg_pos++]);
    std::vector<std::string> data_files;
    for(int i=0; i<num_data_files; i++){
        data_files.push_back(argv[arg_pos++]);
    }
    assert(data_files.size() == num_data_files);
    int eb_mode = atoi(argv[arg_pos++]);
    double eb = atof(argv[arg_pos++]);
    switch(comp_opt){
        case 1:{
            printf("Compress using RBP\n");
            int elements_per_partition = atoi(argv[arg_pos++]);
            umc_compress_prediction_by_regression<T>(reorder_opt, d, elements_per_partition, position_file, conn_file, from_edge, data_files, eb, eb_mode);
            break;
        }
        case 2:{
            printf("Compress using NBP-IDW\n");
            int n_neighbors = atoi(argv[arg_pos++]);
            umc_compress_prediction_by_adjacent_nodes_simple<T>(reorder_opt, position_file, conn_file, from_edge, data_files, n_neighbors, d, eb, eb_mode);
            break;
        }
        case 3:{
            printf("Compress using NBP-KW\n");
            int min_neighbors = atoi(argv[arg_pos++]);
            int max_iter = atoi(argv[arg_pos++]);
            int batch = atof(argv[arg_pos++]);
            umc_compress_prediction_by_adjacent_nodes_kriging<T>(reorder_opt, position_file, conn_file, from_edge, data_files, min_neighbors, max_iter, batch, d, eb, eb_mode);
            break;
        }
        case 4:{
            printf("Compress using NBP-IDW-INP\n");
            int n_neighbors = atoi(argv[arg_pos++]);
            int batch = atof(argv[arg_pos++]);
            umc_compress_prediction_by_adjacent_nodes_dynamic<T>(reorder_opt, position_file, conn_file, from_edge, data_files, n_neighbors, batch, d, eb, eb_mode);
            break;
        }
        case 5:{
            printf("Compress using ADP-1\n");
            int n_neighbors = atoi(argv[arg_pos++]);
            int elements_per_partition = atoi(argv[arg_pos++]);
            umc_compress_blockwise_adaptive1<T>(reorder_opt, d, position_file, conn_file, from_edge, n_neighbors, elements_per_partition, data_files, eb, eb_mode);
            break;
        }
        case 6:{
            printf("Compress using ADP-2\n");
            int n_neighbors = atoi(argv[arg_pos++]);
            int elements_per_partition = atoi(argv[arg_pos++]);
            double blockwise_sampling_ratio = atof(argv[arg_pos++]);
            umc_compress_blockwise_adaptive2<T>(reorder_opt, d, position_file, conn_file, from_edge, data_files, n_neighbors, elements_per_partition, blockwise_sampling_ratio, eb, eb_mode);
            break;
        }
        case 7:{
            printf("Compress using SD\n");
            umc_compress_sd<T>(reorder_opt, d, conn_file, from_edge, data_files, eb, eb_mode);
            break;
        }
        default:{
            printf("No compression\n");
            break;
        }
    };    
}
