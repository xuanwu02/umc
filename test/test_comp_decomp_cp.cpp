#include <iostream>
#include <stdlib.h>
#include <string>
#include <stdio.h>
#include <vector>
#include <cmath>
#include "comp_decomp_cp_preserve.hpp"

using namespace UMC;

int main(int argc, char ** argv){

    using T = float;
    int arg_pos = 1;
    std::string position_file(argv[arg_pos++]);
    std::string conn_file(argv[arg_pos++]);
    int from_edge = atoi(argv[arg_pos++]);
    int d = atoi(argv[arg_pos++]);
    int opt = atoi(argv[arg_pos++]);
    int reorder_opt = atoi(argv[arg_pos++]);    
    switch(opt){
        case 1:{
            printf("Compress using NBP-IDW for Critical Points Preservation\n");
            std::string Data_file(argv[arg_pos++]);
            double max_pwr_eb = atof(argv[arg_pos++]);
            umc_compress_prediction_by_adjacent_nodes_cp_preserve<T>(reorder_opt, position_file, conn_file, from_edge, Data_file, d, max_pwr_eb);
            break;
        }
        case 2:{
            printf("Decompress using NBP-IDW for Critical Points Preservation\n");
            int num_data_files = atoi(argv[arg_pos++]);
            std::vector<std::string> data_files;
            for(int i=0; i<num_data_files; i++){
                data_files.push_back(argv[arg_pos++]);
            }
            std::string compressed_file(argv[arg_pos++]);
            assert(data_files.size() == num_data_files);
            umc_decompress_prediction_by_adjacent_nodes_cp_preserve<T>(reorder_opt, position_file, conn_file, from_edge, compressed_file, data_files, d);
            break;
        }
        default:{
            break;
        }
    };    
}
