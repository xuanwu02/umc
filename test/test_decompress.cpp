#include <iostream>
#include <stdlib.h>
#include <string>
#include <stdio.h>
#include <vector>
#include <cmath>
#include "decompress.hpp"

using namespace UMC;

int main(int argc, char ** argv){

    using T = float;
    int arg_pos = 1;
    std::string data_dir(argv[arg_pos++]);
    std::string var_name(argv[arg_pos++]);
    std::string position_file(data_dir + argv[arg_pos++]);
    std::string conn_file(data_dir + argv[arg_pos++]);
    int from_edge = atoi(argv[arg_pos++]);
    int d = atoi(argv[arg_pos++]);
    int decomp_opt = atoi(argv[arg_pos++]);
    int reorder_opt = atoi(argv[arg_pos++]);
    int num_timestep = atoi(argv[arg_pos++]);
    std::vector<std::string> data_files;
    std::vector<std::string> compressed_files;
    for(int i=0; i<num_timestep; i++){
        data_files.push_back(data_dir + var_name + ".dat." + std::to_string(i));
        compressed_files.push_back(data_files[i] + ".umc");
    }
    assert(data_files.size() == num_timestep);
    switch(decomp_opt){
        case 1:{
            printf("Decompress using RBP\n");
            int elements_per_partition = atoi(argv[arg_pos++]);
            umc_decompress_prediction_by_regression<T>(reorder_opt, d, elements_per_partition, position_file, conn_file, from_edge, compressed_files, data_files);
            break;
        }
        case 2:{
            printf("Decompress using NBP-IDW\n");
            umc_decompress_prediction_by_adjacent_nodes_simple<T>(reorder_opt, d, position_file, conn_file, from_edge, compressed_files, data_files);
            break;
        }
        case 3:{
            printf("Decompress using NBP-KW\n");
            int batch = atof(argv[arg_pos++]);
            umc_decompress_prediction_by_adjacent_nodes_kriging<T>(reorder_opt, d, position_file, conn_file, from_edge, batch, compressed_files, data_files);
            break;
        }
        case 4:{
            printf("Decompress using NBP-IDW-INP\n");
            int batch = atof(argv[arg_pos++]);
            umc_decompress_prediction_by_adjacent_nodes_dynamic<T>(reorder_opt, d, position_file, conn_file, from_edge, batch, compressed_files, data_files);
            break;
        }
        case 6:{
            printf("Decompress using ADP-2\n");
            umc_decompress_blockwise_adaptive2<T>(reorder_opt, d, position_file, conn_file, from_edge, compressed_files, data_files);
            break;
        }
        case 7:{
            printf("Decompress using SD\n");
            umc_decompress_sd<T>(reorder_opt, d, conn_file, from_edge, compressed_files, data_files);
            break;
        }
        default:{
            printf("No decompression\n");
            break;
        }
    };
}