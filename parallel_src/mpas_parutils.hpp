#ifndef _UMC_PARUTILS_HPP
#define _UMC_PARUTILS_HPP

#include <mpi.h>
#include <metis.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <set>
#include "SZ3/api/sz.hpp"
#include "zfp.h"

const std::string varlist_file = "/scratch/user/ocean_mpas/varnames_91.txt";
const std::string part_prefix = "/scratch/user/ocean_mpas/mesh/partition/";
const std::string conn_file = "/scratch/user/ocean_mpas/mesh/conn.dat";
const std::string position_file = "/scratch/user/ocean_mpas/mesh/coords.dat";
const std::string data_file_dir = "/scratch/user/ocean_mpas/";
const std::string compressed_file_dir_idw = "/scratch/user/ocean_mpas/compressed/idw/";
const std::string compressed_file_dir_sz3ori = "/scratch/user/ocean_mpas/compressed/sz3ori/";
const std::string compressed_file_dir_sz3rod = "/scratch/user/ocean_mpas/compressed/sz3rod/";
const std::string compressed_file_dir_zfpori = "/scratch/user/ocean_mpas/compressed/zfpori/";
const std::string compressed_file_dir_zfprod = "/scratch/user/ocean_mpas/compressed/zfprod/";

namespace UMC{

template<typename T>
MPI_Datatype get_mpi_dtype();

template<>
MPI_Datatype get_mpi_dtype<float>(){
    return MPI_FLOAT;
}

template<>
MPI_Datatype get_mpi_dtype<double>(){
    return MPI_DOUBLE;
}

template <class T>
void print_statistics(const T * data_ori, const T * data_dec, size_t data_size){
    double max_val = data_ori[0];
    double min_val = data_ori[0];
    double max_abs = fabs(data_ori[0]);
    for(int i=0; i<data_size; i++){
        if(data_ori[i] > max_val) max_val = data_ori[i];
        if(data_ori[i] < min_val) min_val = data_ori[i];
        if(fabs(data_ori[i]) > max_abs) max_abs = fabs(data_ori[i]);
    }
    double max_err = 0;
    int pos = 0;
    double mse = 0;
    for(int i=0; i<data_size; i++){
        double err = data_ori[i] - data_dec[i];
        mse += err * err;
        if(fabs(err) > max_err){
            pos = i;
            max_err = fabs(err);
        }
    }
    mse /= data_size;
    double psnr = 20 * log10((max_val - min_val) / sqrt(mse));
    std::cout << "Max value = " << max_val << ", min value = " << min_val << ", range = " << max_val - min_val << std::endl;
    std::cout << "Max error = " << max_err  << ", pos = " << pos << std::endl;
    std::cout << "MSE = " << mse << std::endl;
    std::cout << "PSNR = " << psnr << std::endl;
}

template <class T>
std::vector<T> compute_offsets(const std::vector<T>& ref, int len_ref){
    std::vector<T> offset_list(len_ref+1, 0);
    for(int i=1; i<=len_ref; i++){
        offset_list[i] = offset_list[i-1] + ref[i-1];
    }
    return offset_list;
}

template <class T>
std::vector<T> compute_load(const std::vector<T>& dist, int len_dist){
    std::vector<T> local_load(len_dist-1, 0);
    for(int i=0; i<len_dist-1; i++){
        local_load[i] = dist[i+1] - dist[i];
    }
    return local_load;
}

template<class T>
char * SZ3_compress_1D(size_t num_elements, T * data, double abs_eb, size_t& compressed_size){
    SZ3::Config conf(num_elements);
    conf.cmprAlgo = SZ3::ALGO_INTERP_LORENZO;
    conf.errorBoundMode = SZ3::EB_ABS;
    conf.absErrorBound = abs_eb;
    size_t cmpSize = 0;
    char *cmpData = SZ_compress<T>(conf, data, cmpSize);
    compressed_size = cmpSize;
    return cmpData;
}

template<class T>
void SZ3_decompress(char * cmpData, size_t compressed_size, T * dec_data){
    SZ3::Config conf1;
    SZ_decompress<T>(conf1, cmpData, compressed_size, dec_data);
}

unsigned char * zfp_compress_1D(float * array, double tolerance, size_t n1, size_t& out_size){
	int status = 0;
	zfp_type type = zfp_type_float;
	zfp_field* field = zfp_field_1d(array, type, n1);
	zfp_stream* zfp = zfp_stream_open(NULL);
	zfp_stream_set_accuracy(zfp, tolerance);
	size_t bufsize = zfp_stream_maximum_size(zfp, field);
	void* buffer = malloc(bufsize);
	bitstream* stream = stream_open(buffer, bufsize);
	zfp_stream_set_bit_stream(zfp, stream);
	zfp_stream_rewind(zfp);
	size_t zfpsize = zfp_compress(zfp, field);
    if (!zfpsize) {
      fprintf(stderr, "compression failed\n");
      status = 1;
    }	
	zfp_field_free(field);
	zfp_stream_close(zfp);
	stream_close(stream);
	out_size = zfpsize;
	return (unsigned char *)buffer;
}

float * zfp_decompress_1D(unsigned char * comp_data, double tolerance, size_t buffer_size, size_t n1){
	int status = 0;
	float * array = (float *) malloc(n1 * sizeof(float));
	zfp_type type = zfp_type_float;
	zfp_field* field = zfp_field_1d(array, type, n1);
	zfp_stream* zfp = zfp_stream_open(NULL);
	zfp_stream_set_accuracy(zfp, tolerance);
	size_t bufsize = zfp_stream_maximum_size(zfp, field);
	void* buffer = (void *) comp_data;
	bufsize = buffer_size;
	bitstream* stream = stream_open(buffer, bufsize);
	zfp_stream_set_bit_stream(zfp, stream);
	zfp_stream_rewind(zfp);

    if (!zfp_decompress(zfp, field)) {
      fprintf(stderr, "decompression failed\n");
      status = 1;
    }
	zfp_field_free(field);
	zfp_stream_close(zfp);
	stream_close(stream);
	return array;
}

std::vector<std::string> read_varnames(){
    std::vector<std::string> varnames;
    std::ifstream varname_list(varlist_file.c_str());
    std::string line;
    while(std::getline(varname_list, line)){
        varnames.push_back(line);
    }
    varname_list.close();
    return varnames;
}

template<class Type>
std::vector<Type> readfile(const char *file, size_t &num) {
    std::ifstream fin(file, std::ios::binary);
    if (!fin) {
        std::cout << " Error, Couldn't find the file" << "\n";
        return std::vector<Type>();
    }
    fin.seekg(0, std::ios::end);
    const size_t num_elements = fin.tellg() / sizeof(Type);
    fin.seekg(0, std::ios::beg);
    auto data = std::vector<Type>(num_elements);
    fin.read(reinterpret_cast<char *>(&data[0]), num_elements * sizeof(Type));
    fin.close();
    num = num_elements;
    return data;
}

template<class Type>
void writefile(const char *file, Type *data, size_t num_elements) {
    std::ofstream fout(file, std::ios::binary);
    fout.write(reinterpret_cast<const char *>(&data[0]), num_elements * sizeof(Type));
    fout.close();
}

template<class T>
void read(T &var, unsigned char const *&compressed_data_pos) {
    memcpy(&var, compressed_data_pos, sizeof(T));
    compressed_data_pos += sizeof(T);
}

template<class T>
void write(T const var, unsigned char *&compressed_data_pos) {
    memcpy(compressed_data_pos, &var, sizeof(T));
    compressed_data_pos += sizeof(T);
}

template <class T>
inline void read_array_from_src(const unsigned char *& src, T * offset, size_t length){
    if(length > 0){
        memcpy(offset, src, length*sizeof(T));
        src += length * sizeof(T);
    }
}

template <class T>
inline void write_array_to_dst(unsigned char *& dst, const T * array, size_t length){
    if(length > 0){
        memcpy(dst, array, length*sizeof(T));
        dst += length*sizeof(T);
    }
}

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

void init_part_map(const std::vector<idx_t>& part, const std::vector<int32_t>& num_part_nodes, std::vector<uint32_t>& part_map, std::vector<uint32_t>& part_map_inv){
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