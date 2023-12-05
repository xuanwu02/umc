
#ifndef _UMC_CP_PRESERVE_HPP
#define _UMC_CP_PRESERVE_HPP

#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <cmath>
#include "sz_huffman.hpp"

#define MAX_(a, b) ((a>b)?(a):(b))
#define MIN_(a, b) ((a<b)?(a):(b))

namespace UMC{

inline int eb_exponential_quantize(double& eb, const int base, const double log_of_base, const double threshold=std::numeric_limits<float>::epsilon()){
	if(eb <= threshold){
		eb = 0;
		return 0;
	}
	int id = log2(eb / threshold)/log_of_base;
	eb = pow(base, id) * threshold;
	return id;
}

inline int eb_linear_quantize(double& eb, double threshold){
	int id = eb / threshold;
	eb = id * threshold;
	return id;
}

typedef struct Tet{
	int vertex[4];
}Tet;

std::vector<Tet> construct_tets(int n, int m, const int * tets_ind, std::vector<std::vector<std::pair<int, int>>>& point_tets){
	std::vector<Tet> tets;
	point_tets.clear();
	for(int i=0; i<n; i++){
		point_tets.push_back(std::vector<std::pair<int, int>>());
	}
	const int * tets_ind_pos = tets_ind;
	for(int i=0; i<m; i++){
		Tet t;
		for(int j=0; j<4; j++){
			int ind = *(tets_ind_pos ++);
			t.vertex[j] = ind;
			point_tets[ind].push_back(std::make_pair(i, j));
		}
		tets.push_back(t);
	}
	return tets;
}

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

template <typename T> bool same_direction(T u0, T u1, T u2, T u3) {
    int sgn0 = sgn(u0);
    if(sgn0 == 0) return false;
    if((sgn0 == sgn(u1)) && (sgn0 == sgn(u2)) && (sgn0 == sgn(u3))) return true;
    return false;
}

template<typename T>
inline double max_eb_to_keep_sign_3d_online(const T A, const T B, const T C, const T D=0){
	if((A == 0) && (B == 0) && (C == 0)) return 1;
	return fabs(A + B + C + D) / (fabs(A) + fabs(B) + fabs(C));
}

template<typename T>
double 
max_eb_to_keep_position_and_type_3d_online(const T u0, const T u1, const T u2, const T u3, const T v0, const T v1, const T v2, const T v3,
	const T w0, const T w1, const T w2, const T w3){
	double u3_0 = - u3*v1*w2 + u3*v2*w1, u3_1 = - u3*v2*w0 + u3*v0*w2, u3_2 = - u3*v0*w1 + u3*v1*w0;
	double v3_0 = u1*v3*w2 - u2*v3*w1, v3_1 = u2*v3*w0 - u0*v3*w2, v3_2 = u0*v3*w1 - u1*v3*w0;
	double w3_0 = - u1*v2*w3 + u2*v1*w3, w3_1 = u0*v2*w3 - u2*v0*w3, w3_2 = - u0*v1*w3 + u1*v0*w3;
	double c_4 = u0*v1*w2 - u0*v2*w1 + u1*v2*w0 - u1*v0*w2 + u2*v0*w1 - u2*v1*w0;
	double M0 = u3_0 + v3_0 + w3_0;
	double M1 = u3_1 + v3_1 + w3_1;
	double M2 = u3_2 + v3_2 + w3_2;
	double M3 = c_4;
	double M = M0 + M1 + M2 + M3;
	if(M == 0){
		if(same_direction(u0, u1, u2, u3) || same_direction(v0, v1, v2, v3) || same_direction(w0, w1, w2, w3)) return 1;
		return 0;
	}
	bool flag[4];
	flag[0] = (M0 == 0) || (M / M0 > 1);
	flag[1] = (M1 == 0) || (M / M1 > 1);
	flag[2] = (M2 == 0) || (M / M2 > 1);
	flag[3] = (M3 == 0) || (M / M3 > 1);
	if(flag[0] && flag[1] && flag[2] && flag[3]){
		// cp found
		return 0;
		double eb = 1;
		double cur_eb = 0;
		eb = MIN_(eb, max_eb_to_keep_sign_3d_online(u3_0, v3_0, w3_0));
		eb = MIN_(eb, max_eb_to_keep_sign_3d_online(u3_1, v3_1, w3_1));
		eb = MIN_(eb, max_eb_to_keep_sign_3d_online(u3_2, v3_2, w3_2));
		eb = MIN_(eb, max_eb_to_keep_sign_3d_online(u3_0 + u3_1 + u3_2, v3_0 + v3_1 + v3_2, w3_0 + w3_1 + w3_2));
		return eb;
	}
	else{
		double eb = 0;
		double cur_eb = 0;
		if(!flag[0]){
			cur_eb = MIN_(max_eb_to_keep_sign_3d_online(u3_0, v3_0, w3_0), 
					max_eb_to_keep_sign_3d_online(u3_1 + u3_2, v3_1 + v3_2, w3_1 + w3_2, c_4));
			eb = MAX_(eb, cur_eb);
		}
		if(!flag[1]){
			cur_eb = MIN_(max_eb_to_keep_sign_3d_online(u3_1, v3_1, w3_1), 
					max_eb_to_keep_sign_3d_online(u3_0 + u3_2, v3_0 + v3_2, w3_0 + w3_2, c_4));
			eb = MAX_(eb, cur_eb);
		}
		if(!flag[2]){
			cur_eb = MIN_(max_eb_to_keep_sign_3d_online(u3_2, v3_2, w3_2), 
					max_eb_to_keep_sign_3d_online(u3_0 + u3_1, v3_0 + v3_1, w3_0 + w3_1, c_4));
			eb = MAX_(eb, cur_eb);
		}
		if(!flag[3]){
			cur_eb = max_eb_to_keep_sign_3d_online(u3_0 + u3_1 + u3_2, v3_0 + v3_1 + v3_2, w3_0 + w3_1 + w3_2);
			eb = MAX_(eb, cur_eb);
		}
		return eb;
	}
}

template <typename T>
inline void
read_variable_from_src(const unsigned char *& src, T& var){
    memcpy(&var, src, sizeof(T));
    src += sizeof(T);
}

template <typename T>
inline void
write_variable_to_dst(unsigned char *& dst, const T& var){
    memcpy(dst, &var, sizeof(T));
    dst += sizeof(T);
}

template <typename T>
inline void
write_array_to_dst(unsigned char *& dst, const T * array, size_t length){
    memcpy(dst, array, length*sizeof(T));
    dst += length*sizeof(T);
}

HuffmanTree *
build_Huffman_tree(size_t state_num, const int * type, size_t num_elements){
	HuffmanTree * huffman = createHuffmanTree(state_num);
	init(huffman, type, num_elements);
	return huffman;
}

void
Huffman_encode_tree_and_data(size_t state_num, const int * type, size_t num_elements, unsigned char*& compressed_pos){
	HuffmanTree * huffman = build_Huffman_tree(state_num, type, num_elements);
	size_t node_count = 0;
	size_t i = 0;
	for (i = 0; i < state_num; i++){
		if (huffman->code[i]) node_count++;
	}
	node_count = node_count*2-1;
	unsigned char *tree_structure = NULL;
	unsigned int tree_size = convert_HuffTree_to_bytes_anyStates(huffman, node_count, &tree_structure);
	write_variable_to_dst(compressed_pos, node_count);
	write_variable_to_dst(compressed_pos, tree_size);
	write_array_to_dst(compressed_pos, tree_structure, tree_size);
	unsigned char * type_array_size_pos = compressed_pos;
	compressed_pos += sizeof(size_t);
	size_t type_array_size = 0; 
	encode(huffman, type, num_elements, compressed_pos, &type_array_size);
	write_variable_to_dst(type_array_size_pos, type_array_size);
	compressed_pos += type_array_size;
	free(tree_structure);
	SZ_ReleaseHuffman(huffman);
}

int *
Huffman_decode_tree_and_data(size_t state_num, size_t num_elements, const unsigned char *& compressed_pos){
	HuffmanTree* huffman = createHuffmanTree(state_num);
	size_t node_count = 0;
	read_variable_from_src(compressed_pos, node_count);
	unsigned int tree_size = 0;
	read_variable_from_src(compressed_pos, tree_size);
	node root = reconstruct_HuffTree_from_bytes_anyStates(huffman, compressed_pos, node_count);
	compressed_pos += tree_size;
	size_t type_array_size = 0;
	read_variable_from_src(compressed_pos, type_array_size);
	int * type = (int *) malloc(num_elements * sizeof(int));
	decode(compressed_pos, num_elements, root, type);
	compressed_pos += type_array_size;
	SZ_ReleaseHuffman(huffman);
	return type;
}

template<typename T>
T *
log_transform(const T * data, unsigned char * sign, size_t n, bool verbose=false){
    T * log_data = (T *) malloc(n*sizeof(T));
    for(int i=0; i<n; i++){
        sign[i] = 0;
        if(data[i] != 0){
            sign[i] = (data[i] > 0);
            log_data[i] = (data[i] > 0) ? log2f(data[i]) : log2f(-data[i]); 
        }
        else{
            sign[i] = 0;
            log_data[i] = -100; //TODO???
        }
    }
    return log_data;
}

template<typename T>
T *
log_transform(const T * data, unsigned char * sign, unsigned char * zero, size_t n){
    T * log_data = (T *) malloc(n*sizeof(T));
    for(int i=0; i<n; i++){
        sign[i] = 0;
        if(data[i] != 0){
            sign[i] = (data[i] > 0);
            log_data[i] = (data[i] > 0) ? log2(data[i]) : log2(-data[i]); 
            zero[i] = 0;
        }
        else{
            sign[i] = 0;
            log_data[i] = -100; //TODO???
            zero[i] = 1;
        }
    }
    return log_data;
}

}
#endif