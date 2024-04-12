#include <mpi.h>
#include "mpas_parutils.hpp"
#include "mpas_idw.hpp"

using namespace UMC;

int main(int argc, char **argv){
    int num_parts = atoi(argv[1]);
    std::string nparts_str = std::to_string(num_parts);

    int maxEdges = 7, d = 3;
    size_t num = 0;
    auto conn = readfile<int>(conn_file.c_str(), num);
    size_t num_elements = *std::max_element(conn.begin(), conn.end());
    std::vector<std::set<int32_t>> complete_adj_list = generate_adjacent_list_ocean(num_elements, maxEdges, conn);
    idx_t nparts = num_parts;
    std::vector<idx_t> part(num_elements);
    part = (nparts == 1) ? std::vector<idx_t>(num_elements, 0) : create_partition(num_elements, nparts, complete_adj_list);
	std::vector<uint32_t> part_map(num_elements, -1);
	std::vector<uint32_t> part_map_inv(num_elements, -1);
	std::vector<int32_t> num_part_nodes(nparts, 0);
	for(int i=0; i<part.size(); i++){
	    num_part_nodes[part[i]] ++;
	}
	init_part_map(part, num_part_nodes, part_map, part_map_inv);

    std::string part_file = part_prefix + nparts_str + ".part";
    std::string part_map_file = part_prefix + nparts_str + ".part_map";
    std::string num_part_nodes_file = part_prefix + nparts_str + ".num_part_nodes";

    writefile(part_file.c_str(), part.data(), part.size());
    writefile(part_map_file.c_str(), part_map.data(), part_map.size());
    writefile(num_part_nodes_file.c_str(), num_part_nodes.data(), num_part_nodes.size());

    return 0;
}