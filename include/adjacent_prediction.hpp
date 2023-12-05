#ifndef _UMC_ADJ_PRED_HPP
#define _UMC_ADJ_PRED_HPP

#include <vector>
#include <set>
#include <unordered_set>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <cmath>
#include <numeric>
#include "utils.hpp"
#include "partition.hpp"

const double sigma = 2;
const int cc = 1;
const int p = 2;


namespace UMC{

// @param: positions: N x d
// @param: cells: M x (d+1)
template <class T>
std::vector<std::set<T>> 
generate_processed_adjacent_list(int n, int d, const std::vector<T>& cells, int option=0){
	if(option == 1){
		// option == 1, all edges
		std::cout << d << " " << cells.size() << std::endl;
		assert(cells.size() % 2 == 0);
		int m = cells.size() / 2;
		// init the adjcency list with prior vertices
	    std::vector<std::set<int32_t>> adj_list;
	    for(int i=0; i<n; i++){
	    	adj_list.push_back(std::set<int32_t>());
	    }
	    for(int i=0; i<m; i++){
	    	auto id_j = cells[i*2];
	    	auto id_k = cells[i*2+1];
	    	if(id_k < id_j) adj_list[id_j].insert(id_k); 
	    	else adj_list[id_k].insert(id_j); 
	    }
		return adj_list;
	}
	else{
		// option == 0, all tets
		std::cout << d << " " << cells.size() << " " << cells.size() % (d+1) << std::endl;
		assert(cells.size() % (d+1) == 0);
		int m = cells.size() / (d+1);
		// init the adjcency list with prior vertices
	    std::vector<std::set<T>> adj_list;
	    for(int i=0; i<n; i++){
	    	adj_list.push_back(std::set<T>());
	    }
	    for(int i=0; i<m; i++){
	    	for(int j=0; j<=d; j++){
	    		auto id_j = cells[i*(d+1)+j];
	    		// compute adjacent processed nodes for id_j
	    		for(int k=0; k<=d; k++){
	    			auto id_k = cells[i*(d+1)+k];
	    			if(id_k < id_j){
	    				adj_list[id_j].insert(id_k);
	    			}
	    		}
	    	}
	    }
	    if(option == 2){
	    	// option == 2, for les data, read wedges
	    	size_t num;
	    	auto conn = readfile<int>("prism_cells.dat", num);
	    	assert(conn.size() % 6 == 0);
	    	for(int i=0; i<num; i+=6){
	    		auto id_0 = conn[i];
	    		auto id_1 = conn[i+1];
	    		auto id_2 = conn[i+2];
	    		if(id_1 < id_0) adj_list[id_0].insert(id_1);
	    		else adj_list[id_1].insert(id_0);
	    		if(id_2 < id_0) adj_list[id_0].insert(id_2);
	    		else adj_list[id_2].insert(id_0);
	    		if(id_2 < id_1) adj_list[id_1].insert(id_2);
	    		else adj_list[id_2].insert(id_1);
	    		auto id_3 = conn[i+3];
	    		auto id_4 = conn[i+4];
	    		auto id_5 = conn[i+5];
	    		if(id_4 < id_3) adj_list[id_3].insert(id_4);
	    		else adj_list[id_4].insert(id_3);
	    		if(id_5 < id_3) adj_list[id_3].insert(id_5);
	    		else adj_list[id_5].insert(id_3);
	    		if(id_5 < id_3) adj_list[id_4].insert(id_5);
	    		else adj_list[id_5].insert(id_4);
	    		// between two trianges
	    		if(id_3 < id_0) adj_list[id_0].insert(id_3);
	    		else adj_list[id_3].insert(id_0);
	    		if(id_4 < id_1) adj_list[id_1].insert(id_4);
	    		else adj_list[id_4].insert(id_1);
	    		if(id_5 < id_2) adj_list[id_2].insert(id_5);
	    		else adj_list[id_5].insert(id_2);
	    	}
	    }
		return adj_list;
	}
}

template<class T>
std::vector<std::set<int32_t>> 
generate_adjacent_list(int n, int d, const std::vector<T>& cells, int option=0){
	if(option == 1){
		// option == 1, all edges
		std::cout << d << " " << cells.size() << std::endl;
		assert(cells.size() % 2 == 0);
		int m = cells.size() / 2;
		// init the adjcency list with prior vertices
	    std::vector<std::set<int32_t>> adj_list;
	    for(int i=0; i<n; i++){
	    	adj_list.push_back(std::set<int32_t>());
	    }
	    for(int i=0; i<m; i++){
	    	auto id_j = cells[i*2];
	    	auto id_k = cells[i*2+1];
	    	adj_list[id_j].insert(id_k); 
	    	adj_list[id_k].insert(id_j); 
	    }
		return adj_list;
	}
	else{
		// option == 0, all tets
		std::cout << d << " " << cells.size() << " " << cells.size() % (d+1) << std::endl;
		assert(cells.size() % (d+1) == 0);
		int m = cells.size() / (d+1);
		// init the adjcency list with prior vertices
	    std::vector<std::set<int32_t>> adj_list;
	    for(int i=0; i<n; i++){
	    	adj_list.push_back(std::set<int32_t>());
	    }
	    for(int i=0; i<m; i++){
	    	for(int j=0; j<=d; j++){
	    		auto id_j = cells[i*(d+1)+j];
	    		// compute adjacent processed nodes for id_j
	    		for(int k=0; k<=d; k++){
	    			if(j != k){
		    			auto id_k = cells[i*(d+1)+k];
						if(id_j != id_k) adj_list[id_j].insert(id_k);    				
	    			}
	    		}
	    	}
	    }
	    if(option == 2){
	    	// option == 2, for les data, read wedges
	    	size_t num;
	    	auto conn = readfile<int>("prism_cells.dat", num);
	    	assert(conn.size() % 6 == 0);
	    	for(int i=0; i<num; i+=6){
	    		auto id_0 = conn[i];
	    		auto id_1 = conn[i+1];
	    		auto id_2 = conn[i+2];
	    		adj_list[id_0].insert(id_1);
	    		adj_list[id_0].insert(id_2);
	    		adj_list[id_1].insert(id_0);
	    		adj_list[id_1].insert(id_2);
	    		adj_list[id_2].insert(id_0);
	    		adj_list[id_2].insert(id_1);
	    		auto id_3 = conn[i+3];
	    		auto id_4 = conn[i+4];
	    		auto id_5 = conn[i+5];
	    		adj_list[id_3].insert(id_4);
	    		adj_list[id_3].insert(id_5);
	    		adj_list[id_4].insert(id_4);
	    		adj_list[id_4].insert(id_5);
	    		adj_list[id_5].insert(id_4);
	    		adj_list[id_5].insert(id_5);
	    		// between two trianges
	    		adj_list[id_0].insert(id_3);
	    		adj_list[id_1].insert(id_4);
	    		adj_list[id_2].insert(id_5);
	    		adj_list[id_3].insert(id_0);
	    		adj_list[id_4].insert(id_1);
	    		adj_list[id_5].insert(id_2);
	    	}
	    }
		return adj_list;
	}
}

template <class T>
std::vector<std::set<T>>
regenerate_processed_adjacent_list(int n, const std::vector<std::set<T>>& global_adj_list, const std::vector<uint32_t>& part_map){
    std::vector<std::set<T>> modified_adj_list(global_adj_list);
    for(int i = 0; i < n; i++){
        auto it = modified_adj_list[i].begin();
        while(it != modified_adj_list[i].end()){
            auto& neighbor = *it;
            if(part_map[neighbor] >= part_map[i]){
                it = modified_adj_list[i].erase(it);
            }else{
                ++it;
            }
        }
    }
    return modified_adj_list;
}

std::vector<std::set<int32_t>> 
generate_sample_adjacent_list(int d, const std::vector<int32_t>& sampling_nodes, const std::vector<std::set<int32_t>>& adj_list){
    size_t sample_size = sampling_nodes.size();
	std::cout << " sample_size = " << sample_size << std::endl;
	std::vector<std::set<int32_t>> sample_adj_list(sample_size);
	std::unordered_set<int32_t> sampling_set(sampling_nodes.begin(), sampling_nodes.end());
    for(int i=0; i<sample_size; i++){
		sample_adj_list.push_back(std::set<int32_t>());
		auto id = sampling_nodes[i];
		for(auto& item:adj_list[id]){
			if(sampling_set.find(item) != sampling_set.end()){
                auto it = std::find(sampling_nodes.begin(), sampling_nodes.end(), item);
                if (it != sampling_nodes.end()) {
                    int pos = std::distance(sampling_nodes.begin(), it);
                    sample_adj_list[i].insert(pos);
                }
			}
		}
	}
	return sample_adj_list;
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
    double krg_weight;
	AdjNode(int id_, T weight_, double krg_weight_) : id(id_), weight(weight_), krg_weight(krg_weight_){}
};

template <class T>
bool sortByWeight(const std::pair<int, T>& a, const std::pair<int, T>& b){
	return a.second > b.second;
}

template <class T>
std::vector<std::vector<AdjNode<T>>>
generate_adjacent_neighbors(size_t n, int d, int n_neighbors, const std::vector<std::set<int>>& adj_list, const std::vector<T>& positions){
    std::vector<std::vector<AdjNode<T>>> adj_nodes;
	for(int i=0; i<n; i++){
		adj_nodes.push_back(std::vector<AdjNode<T>>());
	}
	for(int i=0; i<n; i++){
		if(adj_list[i].size() > 0){
			std::vector<std::pair<int, double>> neighbors;
			for(const auto& id:adj_list[i]){
				double wt = 1.0 / distance_euclid(d, &positions[i*d], &positions[id*d]);
				neighbors.push_back(std::make_pair(id, wt));
			}
			std::sort(neighbors.begin(), neighbors.end(), sortByWeight<T>);
			double denominator = 0;
			int num = (n_neighbors < neighbors.size()) ? n_neighbors : neighbors.size();
			for(int j=0; j<num; j++){
				denominator += neighbors[j].second;
			}
			for(int j=0; j<num; j++){
				neighbors[j].second /= denominator;
				adj_nodes[i].push_back(AdjNode(neighbors[j].first, (T)(neighbors[j].second), 0.0));
			}
		}
	}
	return adj_nodes;
}

template <class T>
std::vector<std::vector<AdjNode<T>>>
generate_adjacent_neighbors(size_t n, int d, const std::vector<std::set<int>>& adj_list, const std::vector<T>& positions){
    std::vector<std::vector<AdjNode<T>>> adj_nodes;
	for(int i=0; i<n; i++){
		adj_nodes.push_back(std::vector<AdjNode<T>>());
	}
	for(int i=0; i<n; i++){
		if(adj_list[i].size() > 0){
			std::vector<std::pair<int, double>> neighbors;
			for(const auto& id:adj_list[i]){
				double wt = 1.0 / distance_euclid(d, &positions[i*d], &positions[id*d]);
				neighbors.push_back(std::make_pair(id, wt));
			}
			std::sort(neighbors.begin(), neighbors.end(), sortByWeight<T>);
			double denominator = 0;
			for(auto& item:neighbors){
				denominator += item.second;
			}
			for(int j=0; j<adj_list[i].size(); j++){
				neighbors[j].second /= denominator;
				adj_nodes[i].push_back(AdjNode(neighbors[j].first, (T)(neighbors[j].second), 0.0));
			}
		}
	}
	return adj_nodes;
}

template <class T>
std::vector<std::vector<AdjNode<T>>>
generate_adjacent_neighbors2(size_t n, int d, int n_neighbors, const std::vector<std::set<int>>& adj_list, const std::vector<T>& positions){
    std::vector<std::vector<AdjNode<T>>> adj_nodes;
	for(int i=0; i<n; i++){
		adj_nodes.push_back(std::vector<AdjNode<T>>());
	}
	for(int i=0; i<n; i++){
		if(adj_list[i].size() > 0){
			std::vector<std::pair<int, double>> neighbors;
			for(const auto& id:adj_list[i]){
				double wt = 1.0 / distance_euclid(d, &positions[i*d], &positions[id*d]);
				neighbors.push_back(std::make_pair(id, wt));
			}
			std::sort(neighbors.begin(), neighbors.end(), sortByWeight<T>);
			int num = (n_neighbors < neighbors.size()) ? n_neighbors : neighbors.size();
			for(int j=0; j<num; j++){
				adj_nodes[i].push_back(AdjNode(neighbors[j].first, (T)(neighbors[j].second), 0.0));
			}
		}
	}
	return adj_nodes;
}

template <class T>
std::vector<std::vector<AdjNode<T>>>
generate_adjacent_neighbors2(size_t n, int d, const std::vector<std::set<int>>& adj_list, const std::vector<T>& positions){
    std::vector<std::vector<AdjNode<T>>> adj_nodes;
	for(int i=0; i<n; i++){
		adj_nodes.push_back(std::vector<AdjNode<T>>());
	}
	for(int i=0; i<n; i++){
		if(adj_list[i].size() > 0){
			std::vector<std::pair<int, double>> neighbors;
			for(const auto& id:adj_list[i]){
				double wt = 1.0 / distance_euclid(d, &positions[i*d], &positions[id*d]);
				neighbors.push_back(std::make_pair(id, wt));
			}
			std::sort(neighbors.begin(), neighbors.end(), sortByWeight<T>);
			for(int j=0; j<neighbors.size(); j++){
				adj_nodes[i].push_back(AdjNode(neighbors[j].first, (T)(neighbors[j].second), 0.0));
			}
		}
	}
	return adj_nodes;
}


template <class T>
void update_processed_neighbors_of_changes_nodes(int num_elements, const std::vector<std::set<int32_t>>& original_adj_list, const std::vector<std::vector<AdjNode<T>>>& ref_processed_adj_nodes,
                            						const unsigned char * const& changed_invalid, const unsigned char * const& is_invalid, std::vector<std::vector<AdjNode<T>>>& current_processed_adj_nodes){
    for(int i = 0; i < num_elements; i++){
        if(changed_invalid[i]){
			// find all neighbors of a changed node
            for(auto neighbor_id : original_adj_list[i]){
				if(neighbor_id > i){
					// update neighbors
					if(is_invalid[i]){
						int my_index_in_neighbor = -1;
						for(int j=0; j<current_processed_adj_nodes[neighbor_id].size();j++){
							if(i == current_processed_adj_nodes[neighbor_id][j].id){
								my_index_in_neighbor = j;
								break;
							}
						}
						if(my_index_in_neighbor != -1) current_processed_adj_nodes[neighbor_id].erase(current_processed_adj_nodes[neighbor_id].begin() + my_index_in_neighbor);
					}
					else{
						if(!is_invalid[neighbor_id]){
							int my_index_in_neighbor = 0;
							for(int j=0; j<ref_processed_adj_nodes[neighbor_id].size();j++){
								if(i == ref_processed_adj_nodes[neighbor_id][j].id){
									my_index_in_neighbor = j;
									break;
								}
							}
							current_processed_adj_nodes[neighbor_id].push_back(ref_processed_adj_nodes[neighbor_id][my_index_in_neighbor]);
						}
					}
				}
            }
			// update the node itself
			if(!is_invalid[i]){
				current_processed_adj_nodes[i] = std::vector<AdjNode<T>>();
				for(int j=0; j<ref_processed_adj_nodes[i].size(); j++){
					if((!is_invalid[ref_processed_adj_nodes[i][j].id])){
						current_processed_adj_nodes[i].push_back(ref_processed_adj_nodes[i][j]);
					}
				}
			}
        }
    }
}


void init_pred_order_map(const int num_elements, const int stride, std::vector<uint32_t>& pred_order_map, std::vector<uint32_t>& pred_order_map_inv){
    int n_complete_phase = (int) num_elements / stride;
	int complete_part = n_complete_phase * stride;
    int residual = num_elements - complete_part;

	for(int i=0; i<n_complete_phase; i++){
		for(int j=0; j<stride; j++){
			pred_order_map[i*stride + j] = i + j*n_complete_phase;
		}
	}
	if(residual > 0){
		for(int i=complete_part; i<num_elements; i++){
			pred_order_map[i] = i;
		}
	}

    for(int i=0; i<num_elements; i++){
        pred_order_map_inv[pred_order_map[i]] = i;
    }
}

}
#endif
