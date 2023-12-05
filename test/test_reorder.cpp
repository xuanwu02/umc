#include <vector>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <cmath>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <string>
#include <cstdio>
#include "utils.hpp"
#include "reorder.hpp" 
#include "adjacent_prediction.hpp"

using namespace UMC;

int main(int argc, char **argv){

    using T = float;
    int arg_pos = 1;
    std::string position_file(argv[arg_pos++]);
    std::string connectivity_file(argv[arg_pos++]);
    int from_edge = atoi(argv[arg_pos++]);
    int d = atoi(argv[arg_pos++]);
    int reorder_opt = atoi(argv[arg_pos++]);
    int num_data_files = atoi(argv[arg_pos++]);
    std::string data_head(argv[arg_pos++]);
    std::vector<std::string> data_files;
    for(int i=0; i<num_data_files; i++){
        data_files.push_back(argv[arg_pos++]);
    }
    size_t num = 0;
    auto positions = readfile<T>(position_file.c_str(), num);
    assert(num % d == 0);
    size_t num_elements = num / d;
    printf("num_elements = %zu\n", num_elements);
    auto conn = readfile<int>(connectivity_file.c_str(), num);
    auto adj_list = generate_adjacent_list(num_elements, d, conn, from_edge);
    std::vector<int32_t> index_map;
    std::string suffix = ".rod";
    switch(reorder_opt){
        case 1:
            printf("Reorder using DPFS\n");
            index_map = DPFS(adj_list);
            suffix += ".dpfs";
            break;
        case 2:
            printf("Reorder using DFS\n");
            index_map = DFS(adj_list);
            suffix += ".dfs";
            break;
        case 3:
            printf("Reorder using BFS\n");
            index_map = BFS(adj_list);
            suffix += ".bfs";
            break;
        case 4:
            printf("Reorder using BPFS\n");
            index_map = BPFS(adj_list);
            suffix += ".bpfs";
            break;
        default:
            printf("Skipped reordering\n");
            break;
    };
    auto reordered_position = reorder(positions, index_map, num_elements, d);
    auto reordered_conn = reorder_conn(conn, index_map, d);
    writefile((position_file + suffix).c_str(), reordered_position.data(), reordered_position.size());
    writefile((connectivity_file + suffix).c_str(), reordered_conn.data(), reordered_conn.size());
    for(int i=0; i<data_files.size(); i++){
        size_t num_elements = 0;        
        auto data = readfile<T>(data_files[i].c_str(), num_elements);
        auto reordered_data = reorder(data, index_map, num_elements, 1);
        writefile((data_files[i] + suffix).c_str(), reordered_data.data(), reordered_data.size());
    }
}
