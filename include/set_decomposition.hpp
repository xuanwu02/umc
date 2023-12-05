#ifndef _UMC_SET_DECOMPOSITION_HPP
#define _UMC_SET_DECOMPOSITION_HPP

#include <iostream>
#include <stdio.h>
#include <vector>
#include <stack>
#include <set>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <cmath>
#include "utils.hpp"
#include "adjacent_prediction.hpp"

namespace UMC{

// Implementation for Fast and Effective Lossy Compression Algorithms for Scientific Datasets
// "SD" primarily referred to as "RBD2"

template<typename T>
bool sortByValue(const std::pair<int32_t, T>& node1, const std::pair<int32_t, T>& node2) {
    return node1.second < node2.second;
}

template<typename T>
std::vector<std::set<int32_t>>
generate_sbd1_sets(int n, const std::vector<std::pair<int32_t, T>>& sorted_nodes, double eb){
    // store original index of nodes
    // RBD2 step 1: SBD1
    std::vector<std::set<int32_t>> Vs;
    int offset = 0;
    int count = 0;
    while(offset < n){
        std::set<int32_t> Vi;
        for(int j=offset; j<n; j++){
            T diff = fabs(sorted_nodes[j].second - sorted_nodes[offset].second);
            if(diff <= eb){
                auto index = sorted_nodes[j].first;
                Vi.insert(index);
            }
            if(diff > eb) break;
        }
        Vs.push_back(Vi);
        count ++;
        offset += Vi.size();
    }
    assert(offset == n);
    return Vs;
}

void dfs(const std::set<int32_t>& Vi, const std::vector<std::set<int32_t>>& adj_list, std::vector<int>& regionIds, const int startId, int& regionId){
    const int rec = regionId;
    std::stack<int> stack;
    stack.push(startId);
    regionIds[startId] = regionId;
    while(!stack.empty()){
        auto currId = stack.top();
        stack.pop();
        for(auto& neighborId : adj_list[currId]){
            if(Vi.count(neighborId) && regionIds[neighborId] == -1){ // if neighborId is in Vi (i-th sbd1_set) and unvisited
                regionIds[neighborId] = regionId;
                stack.push(neighborId);
            }
        }
    }
    regionId = rec;
}

void init_local_regions_map(const std::set<int32_t>& Vi, const std::vector<std::set<int32_t>>& adj_list, std::vector<int>& regionIds, int& currRegionId){
    for(auto& node:Vi){
        if(regionIds[node] == -1){
            dfs(Vi, adj_list, regionIds, node, currRegionId);
            currRegionId++;
        }
    }
}

void init_rbd2_regions_map(const std::vector<std::set<int32_t>>& sbd1_sets, const std::vector<std::set<int32_t>>& adj_list, std::vector<int>& regionIds){
    // RBD2 step 2: run RBD1 on each sbd1_set
    int currRegionId = 0;
    for(int i=0; i<sbd1_sets.size(); i++){
        init_local_regions_map(sbd1_sets[i], adj_list, regionIds, currRegionId);
    }
}

template<typename T>
std::vector<T> generate_region_mean(int n, int num_regions, const std::vector<int>& regionIds, const std::vector<T>& data){
    std::vector<int> num_region_nodes(num_regions, 0);
    std::vector<double> region_mean(num_regions, 0);
    for(int i=0; i<n; i++){
        num_region_nodes[regionIds[i]]++;
        region_mean[regionIds[i]] += data[i];
    }
    for(int k=0; k<num_regions; k++){
        T mean = region_mean[k] / static_cast<T>(num_region_nodes[k]);
        region_mean[k] = mean;
    }
    std::vector<T> region_mean_T(num_regions, 0);
    for(int i=0; i<num_regions; i++){
        region_mean_T[i] = region_mean[i];
    }
    return region_mean_T;
}

std::vector<int32_t> split_and_generate_NB(int n, int num_regions, const std::vector<std::set<int32_t>>& adj_list,
                                                const std::vector<int>& regionIds, std::vector<std::set<int32_t>>& region_boundary,
                                                    std::vector<std::set<int32_t>>& region_interior, std::vector<char>& visited){
    std::vector<int32_t> N_B(num_regions, 0);
    // for any data i => check its region k => check neighbor of i 
    for(int i=0; i<n; i++){
        bool is_boundary = false;
        auto k = regionIds[i];
        for(auto& j:adj_list[i]){
            if(regionIds[j] != k){
                region_boundary[k].insert(i);
                N_B[k]++;
                visited[i] = true; // mark all boundary nodes as visited
                is_boundary = true;
                break;
            }
        }
        if(!is_boundary) region_interior[k].insert(i);        
    }
    return N_B;
}

std::vector<int32_t> select_seeds_and_generate_NI(int num_regions, const std::vector<std::set<int32_t>>& adj_list, std::vector<char>& visited,
                                                    std::vector<std::set<int32_t>>& region_interior, std::vector<std::set<int32_t>>& region_seeds){
    std::vector<int32_t> N_I(num_regions, 0);
    for(int k=0; k<num_regions; k++){
        if(!region_interior[k].empty()){
            std::set<int>::const_iterator it(region_interior[k].begin());
            std::advance(it, std::rand() % region_interior[k].size());
            auto startSeed = *it;
            // auto startSeed = *region_interior[k].begin();
            std::stack<int> stack;
            stack.push(startSeed);
            visited[startSeed] = true;
            region_seeds[k].insert(startSeed);
            N_I[k]++;
            while(!stack.empty()){
                auto currSeed = stack.top();
                stack.pop();
                bool found_next = false;
                for(auto& neighbor:adj_list[currSeed]){
                    if(!visited[neighbor]){ // if an unvisited interior node is found
                        visited[neighbor] = true;
                        stack.push(neighbor);
                        found_next = true;
                        break;
                    }
                }
                if(!found_next){
                    for(auto& x:region_interior[k]){
                        if(!visited[x]){
                            visited[x] = true;
                            stack.push(x);
                            region_seeds[k].insert(x);
                            N_I[k]++;
                            break;
                        }
                    }
                }
            }
        }
    }
    return N_I;
}

void differential_encoder(int num_regions, const std::vector<std::set<int32_t>>& region_boundary, const std::vector<std::set<int32_t>>& region_seeds,
                            std::vector<int32_t>& N_B, std::vector<int32_t>& N_I, std::vector<int32_t>& I_B, std::vector<int32_t>& I_I){
    for(int k=0; k<num_regions; k++){
        // process boundary
        if(N_B[k] > 0){
            std::vector<int32_t> boundary_k(N_B[k]);
            boundary_k.assign(region_boundary[k].begin(), region_boundary[k].end());
            auto prev = boundary_k.front();
            I_B.push_back(prev);
            if(N_B[k] > 1){
                for(int j=1; j<N_B[k]; j++){
                    auto curr = boundary_k[j];
                    boundary_k[j] -= prev;
                    I_B.push_back(boundary_k[j]);
                    prev = curr;
                }
            }
        }
        // process seeds
        if(N_I[k] > 0){
            std::vector<int32_t> seeds_k(N_I[k]);
            seeds_k.assign(region_seeds[k].begin(), region_seeds[k].end());
            auto prev_ = seeds_k.front();
            I_I.push_back(prev_);
            if(N_I[k] > 1){
                for(int j=1; j<N_I[k]; j++){
                    auto curr_ = seeds_k[j];
                    seeds_k[j] -= prev_;
                    I_I.push_back(seeds_k[j]);
                    prev_ = curr_;
                }
            }
        }
    }
}

template<class T>
void recover_boundary_and_seeds(std::vector<T>& data_recon, std::vector<char>& visited, const std::vector<int>& boundary_index, const std::vector<int>& seeds_index,
                                    const std::vector<T>& region_mean, const std::vector<int>& N_B, std::vector<int>& I_B, const std::vector<int>& N_I, std::vector<int>& I_I){
    size_t num_regions = boundary_index.size();
    for(int k=0; k<num_regions; k++){
        auto start = boundary_index[k];
        visited[I_B[start]] = true;
        data_recon[I_B[start]] = region_mean[k];
        for(int j=start+1; j<start+N_B[k]; j++){
            I_B[j] += I_B[j-1];
            visited[I_B[j]] = true;
            data_recon[I_B[j]] = region_mean[k];
        }
    }
    for(int k=0; k<num_regions; k++){
        if(N_I[k]){
            auto start = seeds_index[k];
            visited[I_I[start]] = true;
            data_recon[I_I[start]] = region_mean[k];
            for(int j=start+1; j<start+N_I[k]; j++){
                I_I[j] += I_I[j-1];
                visited[I_I[j]] = true;
                data_recon[I_I[j]] = region_mean[k];
            }
        }
    }
}

template<class T>
void recover_residual(std::vector<T>& data_recon, std::vector<char>& visited, const std::vector<std::set<int32_t>>& adj_list, const std::vector<int>& I_I){
    auto it = I_I.begin();
    auto currSeed = *it;
    while(it != I_I.end()){
        std::stack<int> stack;
        stack.push(currSeed);
        while(!stack.empty()){
            currSeed = stack.top();
            stack.pop();
            for(auto& neighbor:adj_list[currSeed]){
                if(!visited[neighbor]){
                    visited[neighbor] = true;
                    data_recon[neighbor] = data_recon[currSeed];
                    stack.push(neighbor);
                }
            }
        }
        currSeed = *(++it);
    }
}

template<class T>
void write_vector(const std::vector<T>& vec, unsigned char *& dst){
    memcpy(dst, vec.data(), vec.size()*sizeof(T));
    dst += vec.size()*sizeof(T);
}

template<class T>
void read_vector(std::vector<T>& vec, const unsigned char *& dst){
    memcpy(vec.data(), dst, vec.size()*sizeof(T));
    dst += vec.size()*sizeof(T);
}

}
#endif