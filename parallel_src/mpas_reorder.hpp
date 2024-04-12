#ifndef _UMC_MPAS_REORDER_HPP
#define _UMC_MPAS_REORDER_HPP

#include <metis.h>
#include <iostream>
#include <cstdio>
#include <vector>
#include <queue>
#include <set>
#include <stack>
#include <cassert>

namespace UMC{

template<typename T, class Tmap>
std::vector<T> reorder(const std::vector<T>& data, const std::vector<Tmap>& index_map, int n, int d){
	assert(data.size() == n*d);
	std::vector<T> reordered_data(data.size());
	for(int i=0; i<index_map.size(); i++){
		auto new_i = index_map[i];
		for(int j=0; j<d; j++){
			reordered_data[new_i*d + j] = data[i*d + j];
		}
	}
	return reordered_data;
}

template<typename T, class Tmap>
std::vector<T> reorder_back(const std::vector<T>& reordered_data, const std::vector<Tmap>& reorder_map, int n, int d){
	assert(reordered_data.size() == n*d);
	fflush(stdout);
	std::vector<T> data_before_reorder(reordered_data.size());
	for(int i=0; i<reorder_map.size(); i++){
		auto new_i = reorder_map[i];
		for(int j=0; j<d; j++){
			data_before_reorder[i*d+j] = reordered_data[new_i*d+j];
		}
	}
	return data_before_reorder;
}

template<typename T, class Tmap>
std::vector<std::set<T>> MPAS_reorder_local_adjacent_list(const std::vector<std::set<T>>& adj_list, const std::vector<Tmap>& local_index_map, size_t local_size){
    std::vector<std::set<T>> reordered_adj_list(local_size);
    for (size_t i=0; i<local_size; i++) {
        auto new_local_id = local_index_map[i];
        std::set<T> new_neighbors;
        for(const auto& neighbor : adj_list[i]){
            new_neighbors.insert(local_index_map[neighbor]);
        }
        reordered_adj_list[new_local_id] = std::move(new_neighbors);
    }
    return reordered_adj_list;
}


// breath-first search
std::vector<int32_t> BFS(const std::vector<std::set<int32_t>>& adj_list){
    int n = adj_list.size();
    std::vector<int32_t> index_map(n, 0);
    std::vector<bool> visited(n, false);
    // TODO: consider multiple connected components
    std::queue<int32_t> node_queue;
    visited[0] = true;
    node_queue.push(0);
    int count = 0;
    while(!node_queue.empty()){
    	auto index = node_queue.front();
    	node_queue.pop();
    	index_map[index] = count ++;
    	for(const auto& iter:adj_list[index]){
    		if(!visited[iter]){
    			node_queue.push(iter);
    			visited[iter] = true;
    		}
    	}
    }
    // printf("BFS: #mapped_index = %d, #total_index = %d\n", count, n);
    return index_map;
}

// breath-first search with priority
std::vector<int32_t> BPFS(const std::vector<std::set<int32_t>>& adj_list){
    int n = adj_list.size();
    std::vector<int32_t> index_map(n, 0);
    std::vector<int32_t> processed_nodes(n, 0);
    std::vector<bool> visited(n, false);
    std::set<int32_t> node_set_in_queue;
    visited[0] = true;
    node_set_in_queue.insert(0);
    int count = 0;
    while(!node_set_in_queue.empty()){
    	int max_num_neighbor = 0;
    	int max_index = 0;
    	// pick the node with the most processed neighbors
    	for(const auto& i : node_set_in_queue){
			if(max_num_neighbor < processed_nodes[i]){
				max_num_neighbor = processed_nodes[i];
				max_index = i;
			}    		
    	}
    	// std::cout << max_index << " -> " << count << std::endl;
    	index_map[max_index] = count ++;
    	// if(count % 10000 == 0){
    	// 	std::cout << count << " / " << n << "\n";
    	// }
    	if(count > n){
    		std::cerr << "exceed limits\n";
    		exit(-1);
    	}
    	node_set_in_queue.erase(max_index);
	    // update processed neighbors
		for(const auto& iter:adj_list[max_index]){
			processed_nodes[iter] ++;
    		if(!visited[iter]){
    			node_set_in_queue.insert(iter);
    			visited[iter] = true;
    		}
		}
    }
    // printf("BPFS: #mapped_index = %d, #total_index = %d\n", count, n);
    return index_map;
}

// depth-first search
std::vector<int32_t> DFS(const std::vector<std::set<int32_t>>& adj_list){
    int n = adj_list.size();
    std::vector<int32_t> index_map(n, -1);
    std::vector<bool> visited(n, false);
    std::stack<int32_t> node_stack;
    int count = 0;
    node_stack.push(0);
    while(!node_stack.empty()){
        auto index = node_stack.top();
        node_stack.pop();
        if(!visited[index]){
            visited[index] = true;
            index_map[index] = count++;
        }
        // if(count % 10000 == 0){
        //     std::cout << count << " / " << n << "\n";
        // }
        for(const auto& i : adj_list[index]){
            if(!visited[i]){
                node_stack.push(i);
            }
        }
    }
    // printf("DFS: #mapped_index = %d, #total_index = %d\n", count, n);
    return index_map;
}

std::vector<int32_t> DPFS(const std::vector<std::set<int32_t>>& adj_list) {
    int n = adj_list.size();
    std::vector<int32_t> index_map(n, -1);
    std::vector<int32_t> processed_nodes(n, 0);
    std::vector<bool> visited(n, false);
    std::stack<int32_t> node_stack;
    int count = 0;

    for (int start_node = 0; start_node < n; ++start_node) {
        if (!visited[start_node]) {
            visited[start_node] = true;
            index_map[start_node] = count++;
            node_stack.push(start_node);

            while (!node_stack.empty()) {
                auto index = node_stack.top();
                node_stack.pop();

                for (const auto& iter : adj_list[index]) {
                    processed_nodes[iter]++;
                }

                int max_num_neighbor = 0;
                int max_index = -1;
                for (const auto& i : adj_list[index]) {
                    if (!visited[i] && max_num_neighbor < processed_nodes[i]) {
                        max_num_neighbor = processed_nodes[i];
                        max_index = i;
                    }
                }

                if (max_index != -1) {
                    node_stack.push(max_index);
                    visited[max_index] = true;
                    index_map[max_index] = count++;
                }
            }
        }
    }
    return index_map;
}

std::vector<int32_t> generate_reorder_index_map(int n, const std::vector<std::set<int32_t>>& adj_list, int opt){
	assert(n == adj_list.size());
	std::vector<int32_t> index_map(n, -1);
	switch(opt){
        case 1:            
            index_map = DPFS(adj_list);
            break;
        case 2:            
            index_map = DFS(adj_list);
            break;
        case 3:
            index_map = BFS(adj_list);
            break;
        case 4:
            index_map = BPFS(adj_list);
            break;
        default:
            // printf("Original order\n");
            break;
	};
	return index_map;
}

}
#endif