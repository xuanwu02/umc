#ifndef _UMC_MPAS_IDW_HPP
#define _UMC_MPAS_IDW_HPP

#include <metis.h>
#include <algorithm>
#include <cmath>
#include <vector>
#include <set>

namespace UMC{

template<class T>
std::vector<std::set<int32_t>> 
MPAS_generate_local_adjacent_list(int proc_id, size_t n, int n_col, const std::vector<T>& lines, const std::vector<idx_t>& part, const std::vector<uint32_t>& part_map, int prefix, size_t local_size){
    std::vector<std::set<int32_t>> adj_list;
	size_t pos = 0;
    for(size_t i=0; i<n; i++){
		if(part[i] == proc_id){
			adj_list.push_back(std::set<int32_t>());
			auto local_index = part_map[i] - prefix;
			assert(local_index == pos);
			for(int j=0; j<n_col; j++){
				auto id_j = lines[i * n_col + j] - 1;
				if(id_j != -1){
					if(id_j != i && part[id_j] == proc_id){
						auto local_neighbor_index = part_map[id_j] - prefix;
						adj_list.back().insert(local_neighbor_index); 
						// adj_list[local_neighbor_index].insert(local_index);
					}
				}
			}
			pos ++;
		}
    }
	assert(adj_list.size() == local_size);
    return adj_list;
}

template <class T>
std::vector<std::set<T>>
MPAS_generate_local_processed_adj_list(const std::vector<std::set<T>>& local_adj_list, size_t local_size){
	std::vector<std::set<T>> processed_local_adj_list(local_adj_list);
    for(size_t i=0; i<local_size; i++){
        for(auto it=processed_local_adj_list[i].begin(); it != processed_local_adj_list[i].end();) {
            if(*it > i){
                it = processed_local_adj_list[i].erase(it);
            }else{
                ++it;
            }
        }
    }
	return processed_local_adj_list;
}


template<class T>
std::vector<std::set<int32_t>> 
generate_adjacent_list_ocean(int n, int n_col, const std::vector<T>& cells) {
    std::vector<std::set<int32_t>> adj_list(n);
    for(int i=0; i<n; i++){
        for(int j=0; j<n_col; j++){
            auto id_j = cells[i * n_col + j];
            if(id_j > 0){
                auto adjusted_id_j = id_j - 1;
                if(adjusted_id_j != i){
                    adj_list[i].insert(adjusted_id_j);
                    adj_list[adjusted_id_j].insert(i);
                }
            }
        }
    }
    return adj_list;
}

template <class T>
std::vector<std::set<T>>
generate_local_processed_adj_list(int proc_id, size_t local_node_count, const std::vector<std::set<T>>& global_adj_list, const std::vector<idx_t>& part,
									const std::vector<uint32_t>& part_map, const std::vector<uint32_t>& part_map_inv, uint32_t offset){
	std::vector<std::set<T>> local_adj_list(local_node_count);
	for(int i=0; i<local_node_count; i++){
		local_adj_list.push_back(std::set<T>());
	}
	for(int i=0; i<local_node_count; i++){
		uint32_t original_id = part_map_inv[offset+i];
		local_adj_list[i] = global_adj_list[original_id];
		auto it = local_adj_list[i].begin();
		while(it != local_adj_list[i].end()){
			auto& neighbor = *it;
			if(part[neighbor] != part[original_id] || part_map[neighbor] > part_map[original_id]){
				it = local_adj_list[i].erase(it);
			}else{
				++it;
			}
		}
	}
	return local_adj_list;
}

template<class T>
inline double 
distance_euclid(int d, const T * first, const T * second) {
	double dist = 0;
	for(int i=0; i<d; i++){
		dist += (first[i] - second[i]) * (first[i] - second[i]);
	}
	return sqrt(dist);
}

template <class T>
struct AdjNode{
	int id;
	T weight;
	AdjNode(int id_, T weight_) : id(id_), weight(weight_){}
};

template <class T>
bool sortByWeight(const std::pair<int, T>& a, const std::pair<int, T>& b){
	return a.second > b.second;
}


template <class T1>
std::vector<std::vector<AdjNode<T1>>>
MPAS_generate_local_adjacent_neighbors(size_t local_size, int d, int max_neighbors, const std::vector<std::set<int>>& local_adj_list, const std::vector<T1>& local_positions){
    std::vector<std::vector<AdjNode<T1>>> adj_nodes(local_size);
	for(size_t i=0; i<local_size; i++){
		if(local_adj_list[i].size() > 0){
			std::vector<std::pair<int, double>> neighbors;
			for(const auto& id:local_adj_list[i]){
				double wt = 1.0 / distance_euclid(d, &local_positions[i*d], &local_positions[id*d]);
				neighbors.push_back(std::make_pair(id, wt));
			}
			std::sort(neighbors.begin(), neighbors.end(), sortByWeight<double>);
			double denominator = 0;
			int max_size = (max_neighbors < neighbors.size()) ? max_neighbors : neighbors.size();
			for(int j=0; j<max_size; j++){
				denominator += neighbors[j].second;
			}
			if(denominator > 0){
				for(int j=0; j<max_size; j++){
					neighbors[j].second /= denominator;
					adj_nodes[i].push_back(AdjNode(neighbors[j].first, (T1)(neighbors[j].second)));
				}
			}
		}
	}
	return adj_nodes;
}



template <class T>
std::vector<std::vector<AdjNode<T>>>
generate_local_adjacent_neighbors(size_t local_node_count, int d, int max_neighbors, const std::vector<std::set<int>>& local_adj_list,
									const std::vector<T>& positions, const std::vector<uint32_t>& part_map_inv, uint32_t offset){
    std::vector<std::vector<AdjNode<T>>> adj_nodes;
	for(int i=0; i<local_node_count; i++){
		adj_nodes.push_back(std::vector<AdjNode<T>>());
	}
	for(int i=0; i<local_node_count; i++){
		if(local_adj_list[i].size() > 0){
			std::vector<std::pair<int, double>> neighbors;
			uint32_t original_id = part_map_inv[offset+i];
			for(const auto& id:local_adj_list[i]){
				double wt = 1.0 / distance_euclid(d, &positions[original_id*d], &positions[id*d]);
				neighbors.push_back(std::make_pair(id, wt));
			}
			std::sort(neighbors.begin(), neighbors.end(), sortByWeight<T>);
			double denominator = 0;
			int num = (max_neighbors < neighbors.size()) ? max_neighbors : neighbors.size();
			for(int j=0; j<num; j++){
				denominator += neighbors[j].second;
			}
			assert(denominator > 0);
			for(int j=0; j<num; j++){
				neighbors[j].second /= denominator;
				adj_nodes[i].push_back(AdjNode(neighbors[j].first, (T)(neighbors[j].second)));
			}
		}
	}
	return adj_nodes;
}

}
#endif