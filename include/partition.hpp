#ifndef _UMC_PARTITION_HPP
#define _UMC_PARTITION_HPP

#include <vector>
#include <set>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <random>
#include <metis.h>

namespace UMC{

// create partition using metis
std::vector<idx_t> create_partition(int n_node, int n_parts, const std::vector<std::set<int32_t>>& adj_list){
    idx_t nverts = n_node;
    idx_t ncon = 1; 
    idx_t nparts = n_parts;
    std::vector<idx_t> xadj, adj;   
    for(int i=0; i<n_node; i++){
        xadj.push_back(adj.size());
        for(const auto& iter:adj_list[i]){
            adj.push_back(iter);
        }
    }
    xadj.push_back(adj.size());
    std::vector<idx_t> part(n_node);
    idx_t objval;
    int rtn = METIS_PartGraphKway(
        &nverts, 
        &ncon, 
        &xadj[0],
        &adj[0],
        NULL, 
        NULL, 
        NULL, 
        &nparts,
        NULL,
        NULL,
        NULL,
        &objval,
        &part[0]);
    return part;   
}

// compute part_map and part_map_inv
void init_part_map(const std::vector<idx_t>& part, const std::vector<uint32_t>& num_part_nodes, std::vector<uint32_t>& part_map, std::vector<uint32_t>& part_map_inv){
	auto nparts = num_part_nodes.size();
	auto n_node = part.size();
	std::vector<uint32_t> part_index(nparts, 0);
	// compute offset of each part
	uint32_t index = 0;
	for(int i=0; i<nparts; i++){
		part_index[i] = index;
		index += num_part_nodes[i];
	}		
	// compute the map and inverse map
	for(int i=0; i<n_node; i++){
		part_map[i] = part_index[part[i]];
		part_map_inv[part_index[part[i]]] = i;
		part_index[part[i]] ++;
	}
}

}
#endif
