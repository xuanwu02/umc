#ifndef _PARUMC_HPP
#define _PARUMC_HPP

#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include "mpas_parutils.hpp"
#include "mpas_reorder.hpp"
#include "mpas_idw.hpp"
#include <SZ3/quantizer/IntegerQuantizer.hpp>
#include <SZ3/encoder/HuffmanEncoder.hpp>
#include <SZ3/lossless/Lossless_zstd.hpp>

namespace UMC{

/* idw compression using PDFS data reordering */
template<class T, class T1>
void par_compress_idw(size_t local_size, const std::vector<T>& local_data, const std::vector<std::vector<AdjNode<T1>>>& processed_adj_neighbors, double eb,
                                double& max_error, double& squared_error, std::vector<int>& var_quant_inds, std::vector<T>& var_unpred_data, size_t& local_unpred_count){
    const int quant_radius = 32768;
    SZ3::LinearQuantizer<T> quantizer = SZ3::LinearQuantizer<T>(eb, quant_radius);
    // squared_error = 0;
    // max_error = 0;
    std::vector<T> local_dec_data(local_data);
    local_unpred_count = 0;

    double time = -MPI_Wtime();
    for(int j=0; j<local_size; j++){
        T pred = 0;
        for(int k=0; k<processed_adj_neighbors[j].size(); k++){
            int ind = processed_adj_neighbors[j][k].id;
            pred += processed_adj_neighbors[j][k].weight * local_dec_data[ind];
        }
        auto quant_ind = quantizer.quantize_and_overwrite(local_dec_data[j], pred);
        if(quant_ind == 0){
            var_unpred_data.push_back(local_data[j]);
            local_unpred_count++;
        }
        var_quant_inds.push_back(quant_ind);

        // double error = local_data[j] - local_dec_data[j];
        // if(fabs(error) > max_error) max_error = fabs(error);
        // squared_error += error*error;
    }
    time = +MPI_Wtime();
}

template<class T, class T1>
void par_compress_idw(int timestep, const std::vector<std::string> varnames, const std::string data_file_prefix, const std::string compressed_file_prefix, size_t n_vars, size_t num_elements, double eb_rel,
                        const std::vector<int32_t>& index_map, const std::vector<uint32_t>& part_map, const std::vector<int32_t>& num_part_nodes, const std::vector<int>& vtxdist,
                            const std::vector<std::vector<AdjNode<T1>>>& processed_adj_neighbors, MPI_Comm comm_val, int verb){
    int n_proc, id;
    MPI_Comm_size(comm_val, &n_proc);
    MPI_Comm_rank(comm_val, &id);

    MPI_Datatype mpi_dtype = get_mpi_dtype<T>();

    double comp_time = 0;
    double time1 = 0;
    size_t local_size = num_part_nodes[id];
    size_t overall_compressed_size = 0;
    size_t num = 0;

    std::vector<double> eb_list(n_vars);
    std::vector<int> unpred_count_per_var(n_vars);
    std::vector<int> overall_quant_inds;
    std::vector<T> overall_unpred_data;
    // std::vector<double> max_error_list(n_vars), mse_list(n_vars), psnr_list(n_vars);

    double local_time = 0;
    // double local_reorder_time = 0, local_quantize_time = 0;
    for(int i=0; i<n_vars; i++){
        std::vector<T> stats(3);
        std::vector<T> local_data(local_size), local_data1(local_size);
        std::string data_file = data_file_prefix + varnames[i];
        std::vector<T> data, data1;
        if(!id){
            data.resize(num_elements);
            data = readfile<T>(data_file.c_str(), num);
            data1 = reorder(data, part_map, num_elements, 1);
            stats[0] = *std::min_element(data.begin(), data.end());
            stats[1] = *std::max_element(data.begin(), data.end());
            stats[2] = stats[1] - stats[0];
        }
        MPI_Bcast(stats.data(), 3, mpi_dtype, 0, comm_val);
        MPI_Scatterv(data1.data(), num_part_nodes.data(), vtxdist.data(), mpi_dtype, local_data.data(), local_size, mpi_dtype, 0, comm_val);

        double eb = eb_rel * stats[2];
        eb_list[i] = eb;

        MPI_Barrier(comm_val);
        local_time -= MPI_Wtime();
        double max_error = 0, squared_error = 0;
        size_t local_unpred_count;

        // local_reorder_time -= MPI_Wtime();
        local_data1 = reorder(local_data, index_map, local_size, 1);
        // local_reorder_time += MPI_Wtime();


        // local_quantize_time -= MPI_Wtime();
        par_compress_idw<T, T1>(local_size, local_data1, processed_adj_neighbors, eb, max_error, squared_error, overall_quant_inds, overall_unpred_data, local_unpred_count);
        // local_quantize_time += MPI_Wtime();

        unpred_count_per_var[i] = local_unpred_count;
        local_time += MPI_Wtime();

        // double var_max_error = 0, var_squared_error = 0;
        // MPI_Reduce(&max_error, &var_max_error, 1, MPI_DOUBLE, MPI_MAX, 0, comm_val);
        // MPI_Reduce(&squared_error, &var_squared_error, 1, MPI_DOUBLE, MPI_SUM, 0, comm_val);
        // max_error_list[i] = var_max_error;
        // double mse = var_squared_error / num_elements;
        // mse_list[i] = mse;
        // psnr_list[i] = 20 * log10(stats[2] / sqrt(mse));
    }

    MPI_Barrier(comm_val);
    time1 = -MPI_Wtime();
    double local_encode_time = -MPI_Wtime();
    unsigned char * compressed = (unsigned char *) malloc(n_vars*local_size*2 * sizeof(T));
    unsigned char * compressed_data_pos = compressed;
    size_t overall_unpred_count = overall_unpred_data.size();
    write(overall_unpred_count, compressed_data_pos);
    write_array_to_dst(compressed_data_pos, (double *)&eb_list[0], n_vars);
    if(overall_unpred_count > 0){
        write_array_to_dst(compressed_data_pos, (int *)&unpred_count_per_var[0], n_vars);
        write_array_to_dst(compressed_data_pos, (T *)&overall_unpred_data[0], overall_unpred_data.size());
    }
    auto encoder = SZ3::HuffmanEncoder<int>();
    encoder.preprocess_encode(overall_quant_inds, 0);
    encoder.save(compressed_data_pos);
    encoder.encode(overall_quant_inds, compressed_data_pos);
    encoder.postprocess_encode();
    auto lossless = SZ3::Lossless_zstd();
    size_t compressed_size = 0;
    unsigned char *lossless_data = lossless.compress(compressed, compressed_data_pos - compressed, compressed_size);
    lossless.postcompress_data(compressed);
    local_encode_time += MPI_Wtime();
    MPI_Barrier(comm_val);
    time1 += MPI_Wtime();

    double comp_local_time = local_time + local_encode_time;
    double max_local_time = 0;
    MPI_Reduce(&comp_local_time, &max_local_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm_val);

    size_t all_compressed_size = 0;
    MPI_Reduce(&compressed_size, &all_compressed_size, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, comm_val);

    std::vector<size_t> each_compressed_size(n_proc, 0);
    each_compressed_size[id] = compressed_size;
    MPI_Allgather(MPI_IN_PLACE, 1, MPI_UNSIGNED_LONG, &each_compressed_size[0], 1, MPI_UNSIGNED_LONG, comm_val);
    auto mpi_offsets = compute_offsets(each_compressed_size, n_proc);
    MPI_Barrier(comm_val);

    MPI_File fh;
    std::string compressed_file = compressed_file_prefix + "step" + std::to_string(timestep) + "_" + std::to_string(n_proc) + "_" + std::to_string(eb_rel);
    MPI_File_open(comm_val, compressed_file.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    if(!id) MPI_File_write_at(fh, 0, each_compressed_size.data(), each_compressed_size.size(), MPI_UNSIGNED_LONG, MPI_STATUS_IGNORE);
    MPI_File_write_at(fh, mpi_offsets[id] + n_proc*sizeof(MPI_UNSIGNED_LONG), lossless_data, compressed_size, MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);
    MPI_Barrier(comm_val);
    MPI_File_close(&fh);
    free(lossless_data);
    
    if(!id){
        double aggregated_cr = n_vars*num_elements*sizeof(T) *1.0 / (all_compressed_size + n_proc*sizeof(MPI_UNSIGNED_LONG));
        // printf("MPI_WTICK = %.16f\n", MPI_Wtick());
        printf("aggregated_compression_ratio = %f\n", aggregated_cr);
        printf("compression_time = %f\n", max_local_time);
        fflush(stdout);
        // if(verb){
        //     for(int i=0; i<n_vars; i++){
        //         printf("%d: eb = %.8f, Max error = %.8f, MSE = %.8f, PSNR = %.8f\n", i, eb_list[i], max_error_list[i], mse_list[i], psnr_list[i]);
        //     }
        // }
    }
}

template<class T, class T1>
void par_compress_idw(const std::vector<std::string> varnames, size_t n_vars, int d, int maxEdges, int timestep,
                        int max_neighbors, MPI_Comm comm_val, double eb_rel, int verb){
    int n_proc, id;
    MPI_Comm_size(comm_val, &n_proc);
    MPI_Comm_rank(comm_val, &id);
    
    size_t num = 0;
    auto complete_positions = readfile<T1>(position_file.c_str(), num);
    size_t num_elements = num / d;
    auto conn = readfile<int>(conn_file.c_str(), num);

    // read partition
    std::string nparts_str = std::to_string(n_proc);
    std::string part_file = part_prefix + nparts_str + ".part";
    std::string part_map_file = part_prefix + nparts_str + ".part_map";
    std::string num_part_nodes_file = part_prefix + nparts_str + ".num_part_nodes";
    auto part = readfile<idx_t>(part_file.c_str(), num);
    auto part_map = readfile<uint32_t>(part_map_file.c_str(), num);
    auto num_part_nodes = readfile<int32_t>(num_part_nodes_file.c_str(), num);
    auto vtxdist = compute_offsets(num_part_nodes, n_proc);

    // prepare local_positions
    size_t local_size = num_part_nodes[id];
    std::vector<T1> local_positions(local_size * d);
    size_t flag = 0;
    for(size_t i=0; i<part.size(); i++){
        if(part[i] == id){
            for(int j=0; j<d; j++){
                local_positions[flag*d+j] = complete_positions[i*d+j];
            }
            flag++;
        }
    }
    assert(flag == local_size);
    complete_positions = std::vector<T1>();

    double max_prep_time = 0;

    double prep_time = 0;
    MPI_Barrier(comm_val);
    prep_time = -MPI_Wtime();

    auto local_adj_list = MPAS_generate_local_adjacent_list(id, num_elements, maxEdges, conn, part, part_map, vtxdist[id], local_size);
    auto index_map = generate_reorder_index_map(local_size, local_adj_list, 1);
    local_positions = reorder(local_positions, index_map, local_size, d);
    auto local_adj_list_reordered = MPAS_reorder_local_adjacent_list(local_adj_list, index_map, local_size);
    auto processed_adj_list = MPAS_generate_local_processed_adj_list(local_adj_list_reordered, local_size);
    auto local_processed_neighbors = MPAS_generate_local_adjacent_neighbors(local_size, d, max_neighbors, processed_adj_list, local_positions);

    prep_time += MPI_Wtime();
    MPI_Reduce(&prep_time, &max_prep_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm_val);

    if(!id){
        printf("preprocessing_time = %f\n", max_prep_time);
        fflush(stdout);
    }

    local_adj_list = std::vector<std::set<int32_t>>();
    local_adj_list_reordered = std::vector<std::set<int32_t>>();
    processed_adj_list = std::vector<std::set<int32_t>>();
    conn = std::vector<int>();
    
    std::string data_file_prefix = data_file_dir + "step_" + std::to_string(timestep) + "/";
    std::string compressed_file_prefix = compressed_file_dir_idw;

    par_compress_idw<T, T1>(timestep, varnames, data_file_prefix, compressed_file_prefix, n_vars, num_elements, eb_rel, index_map, part_map, num_part_nodes, vtxdist, local_processed_neighbors, comm_val, verb);
}


/* idw decompression using PDFS data reordering */
template<class T, class T1>
std::vector<T> par_decompress_idw(size_t local_size, const std::vector<int>& local_quant_inds, const std::vector<T>& local_unpred_data,
                                            const std::vector<std::vector<AdjNode<T1>>>& processed_adj_neighbors, double eb){
    const int quant_radius = 32768;
    SZ3::LinearQuantizer<T> quantizer = SZ3::LinearQuantizer<T>(eb, quant_radius);
    const int * quant_inds_pos = local_quant_inds.data();
    const T * unpred_data_pos = local_unpred_data.data();

    std::vector<T> local_dec_data(local_size);
    for(int j=0; j<local_size; j++){
        T pred = 0;
        for(int k=0; k<processed_adj_neighbors[j].size(); k++){
            int ind = processed_adj_neighbors[j][k].id;
            pred += processed_adj_neighbors[j][k].weight * local_dec_data[ind];
        }
        if(*quant_inds_pos != 0){
            local_dec_data[j] = quantizer.recover(pred, *quant_inds_pos);
        }else{
            local_dec_data[j] = *(unpred_data_pos++);
        }
        quant_inds_pos++;
    }

    return local_dec_data;
}

template<class T, class T1>
void par_decompress_idw(int timestep, const std::vector<std::string> varnames, const std::string data_file_prefix, const std::string compressed_file_prefix, size_t n_vars, size_t num_elements,
                        double eb_rel, const std::vector<int32_t>& index_map, const std::vector<uint32_t>& part_map, const std::vector<int32_t>& num_part_nodes, const std::vector<int>& vtxdist,
                            const std::vector<std::vector<AdjNode<T1>>>& processed_adj_neighbors, MPI_Comm comm_val, int verb){
    int n_proc, id;
    MPI_Comm_size(comm_val, &n_proc);
    MPI_Comm_rank(comm_val, &id);
    
    MPI_Datatype mpi_dtype = get_mpi_dtype<T>();

    double decomp_time = 0;
    double time1;
    size_t local_size = num_part_nodes[id];

    MPI_File fh_read;
    std::string compressed_file = compressed_file_prefix + "step" + std::to_string(timestep) + "_" + std::to_string(n_proc) + "_" + std::to_string(eb_rel);
    MPI_File_open(comm_val, compressed_file.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh_read);
    std::vector<size_t> each_compressed_size(n_proc, -1);
    if(!id) MPI_File_read_at(fh_read, 0, &each_compressed_size[0], n_proc, MPI_UNSIGNED_LONG, MPI_STATUS_IGNORE);
    MPI_Bcast(&each_compressed_size[0], n_proc, MPI_UNSIGNED_LONG, 0, comm_val);
    auto mpi_offsets = compute_offsets(each_compressed_size, n_proc);

    size_t compressed_size = each_compressed_size[id];
    unsigned char * input = (unsigned char *) malloc(compressed_size*sizeof(unsigned char));
    MPI_File_read_at(fh_read, mpi_offsets[id] + n_proc*sizeof(MPI_UNSIGNED_LONG), input, compressed_size, MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);
    MPI_Barrier(comm_val);
    MPI_File_close(&fh_read);

    MPI_Barrier(comm_val);
    time1 = -MPI_Wtime();
    auto lossless = SZ3::Lossless_zstd();
    size_t remaining_length = compressed_size;
    unsigned char * compressed_data = lossless.decompress(input, compressed_size);
    const unsigned char * compressed_data_pos = compressed_data;
    size_t overall_unpred_count = 0;
    read(overall_unpred_count, compressed_data_pos);
    std::vector<double> eb_list(n_vars);
    read_array_from_src(compressed_data_pos, &eb_list[0], n_vars);
    std::vector<int> unpred_count_per_var(n_vars);
    std::vector<T> overall_unpred_data(overall_unpred_count);
    std::vector<int> vars_unpred_data_offsets(n_vars+1, 0);
    if(overall_unpred_count > 0){
        read_array_from_src(compressed_data_pos, &unpred_count_per_var[0], n_vars);
        read_array_from_src(compressed_data_pos, &overall_unpred_data[0], overall_unpred_count);
        vars_unpred_data_offsets = compute_offsets(unpred_count_per_var, n_vars);
    }
    auto encoder = SZ3::HuffmanEncoder<int>();
    encoder.load(compressed_data_pos, remaining_length);
    std::vector<int> overall_quant_inds;
    overall_quant_inds = encoder.decode(compressed_data_pos, n_vars*local_size);
    encoder.postprocess_decode();
    lossless.postdecompress_data(compressed_data);
    time1 += MPI_Wtime();
    decomp_time += time1;

    for(int i=0; i<n_vars; i++){
        MPI_Barrier(comm_val);
        time1 = -MPI_Wtime();

        std::vector<int> var_local_quant_inds(overall_quant_inds.begin()+i*local_size, overall_quant_inds.begin()+(i+1)*local_size);
        std::vector<T> var_local_unpred_data(unpred_count_per_var[i]);
        if(unpred_count_per_var[i] > 0) std::copy(overall_unpred_data.begin()+vars_unpred_data_offsets[i], overall_unpred_data.begin()+vars_unpred_data_offsets[i+1], var_local_unpred_data.begin());
        std::vector<T> var_local_dec_data = par_decompress_idw<T, T1>(local_size, var_local_quant_inds, var_local_unpred_data, processed_adj_neighbors, eb_list[i]);

        time1 += MPI_Wtime();
        decomp_time += time1;

        std::vector<T> var_dec_data(num_elements);
        std::vector<T> interm_data = reorder_back(var_local_dec_data, index_map, local_size, 1);
        MPI_Gatherv(&interm_data[0], local_size, mpi_dtype, &var_dec_data[0], &num_part_nodes[0], &vtxdist[0], mpi_dtype, 0, comm_val);
        
        if(!id){
            std::vector<T> dec_data_original_order;
            dec_data_original_order = reorder_back(var_dec_data, part_map, num_elements, 1);
            std::string data_file = data_file_prefix + varnames[i];
            size_t num = 0;
            std::vector<T> data_ori = readfile<T>(data_file.c_str(), num);
            // if(verb){
            //     std::cout << "variable " << i << ": " << std::endl;
            //     print_statistics(data_ori.data(), dec_data_original_order.data(), num_elements);
            // }
            writefile((data_file+".merge.umc.out").c_str(), dec_data_original_order.data(), num_elements);
        }
    }

    double max_decomp_time = 0;
    MPI_Reduce(&decomp_time, &max_decomp_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm_val);
    if(!id){
        printf("decomprression_time = %f\n", max_decomp_time);
        fflush(stdout);
    }
}

template<class T, class T1>
void par_decompress_idw(const std::vector<std::string> varnames, size_t n_vars, int d, int maxEdges, int timestep, int max_neighbors, MPI_Comm comm_val, double eb_rel, int verb){
    int n_proc, id;
    MPI_Comm_size(comm_val, &n_proc);
    MPI_Comm_rank(comm_val, &id);
    
    size_t num = 0;
    auto complete_positions = readfile<T1>(position_file.c_str(), num);
    size_t num_elements = num / d;
    auto conn = readfile<int>(conn_file.c_str(), num);

    // read partition
    std::string nparts_str = std::to_string(n_proc);
    std::string part_file = part_prefix + nparts_str + ".part";
    std::string part_map_file = part_prefix + nparts_str + ".part_map";
    std::string num_part_nodes_file = part_prefix + nparts_str + ".num_part_nodes";
    auto part = readfile<idx_t>(part_file.c_str(), num);
    auto part_map = readfile<uint32_t>(part_map_file.c_str(), num);
    auto num_part_nodes = readfile<int32_t>(num_part_nodes_file.c_str(), num);
    auto vtxdist = compute_offsets(num_part_nodes, n_proc);

    // prepare local_positions
    size_t local_size = num_part_nodes[id];
    std::vector<T1> local_positions(local_size * d);
    size_t flag = 0;
    for(size_t i=0; i<part.size(); i++){
        if(part[i] == id){
            for(int j=0; j<d; j++){
                local_positions[flag*d+j] = complete_positions[i*d+j];
            }
            flag++;
        }
    }
    assert(flag == local_size);
    complete_positions = std::vector<T1>();

    double prep_time = 0;
    MPI_Barrier(comm_val);
    prep_time -= MPI_Wtime();

    auto local_adj_list = MPAS_generate_local_adjacent_list(id, num_elements, maxEdges, conn, part, part_map, vtxdist[id], local_size);
    auto index_map = generate_reorder_index_map(local_size, local_adj_list, 1);
    local_positions = reorder(local_positions, index_map, local_size, d);
    auto local_adj_list_reordered = MPAS_reorder_local_adjacent_list(local_adj_list, index_map, local_size);
    auto processed_adj_list = MPAS_generate_local_processed_adj_list(local_adj_list_reordered, local_size);
    auto local_processed_neighbors = MPAS_generate_local_adjacent_neighbors(local_size, d, max_neighbors, processed_adj_list, local_positions);

    MPI_Barrier(comm_val);
    prep_time += MPI_Wtime();

    local_adj_list = std::vector<std::set<int32_t>>();
    local_adj_list_reordered = std::vector<std::set<int32_t>>();
    processed_adj_list = std::vector<std::set<int32_t>>();
    conn = std::vector<int>();

    std::string data_file_prefix = data_file_dir + "step_" + std::to_string(timestep) + "/";
    std::string compressed_file_prefix = compressed_file_dir_idw;

    par_decompress_idw<T, T1>(timestep, varnames, data_file_prefix, compressed_file_prefix, n_vars, num_elements, eb_rel, index_map, part_map, num_part_nodes, vtxdist, local_processed_neighbors, comm_val, verb);

    if(!id){
        printf("preprocessing_time = %f\n", prep_time);
        fflush(stdout);
    }
}

/* sz3 compression without data reordering */
template<class T>
void par_compress_sz3ori(int timestep, const std::vector<std::string> varnames, const std::string data_file_prefix, const std::string compressed_file_prefix, size_t n_vars, size_t num_elements, double eb_rel,
                            const std::vector<uint32_t>& part_map, const std::vector<int32_t>& num_part_nodes, const std::vector<int>& vtxdist, MPI_Comm comm_val, int verb){
    int n_proc, id;
    MPI_Comm_size(comm_val, &n_proc);
    MPI_Comm_rank(comm_val, &id);

    MPI_Datatype mpi_dtype = get_mpi_dtype<T>();

    size_t compressed_size = 0;
    size_t local_size = num_part_nodes[id];
    size_t num = 0;

    double local_time = 0;
    for(int i=0; i<n_vars; i++){
        std::vector<size_t> cmpSize_set(n_proc);

        std::vector<T> stats(3);
        std::vector<T> local_data(local_size);
        std::string data_file = data_file_prefix + varnames[i];
        std::vector<T> data, data1;
        if(!id){
            data.resize(num_elements);
            data = readfile<T>(data_file.c_str(), num);
            data1 = reorder(data, part_map, num_elements, 1);
            stats[0] = *std::min_element(data.begin(), data.end());
            stats[1] = *std::max_element(data.begin(), data.end());
            stats[2] = stats[1] - stats[0];
        }
        MPI_Bcast(&stats[0], 3, mpi_dtype, 0, comm_val);
        MPI_Scatterv(&data1[0], &num_part_nodes[0], &vtxdist[0], mpi_dtype, &local_data[0], local_size, mpi_dtype, 0, comm_val);
        
        MPI_Barrier(comm_val);
        local_time -= MPI_Wtime();
        double eb = eb_rel * stats[2];
        size_t cmpSize = 0;
        char * cmpData = SZ3_compress_1D(local_size, local_data.data(), eb, cmpSize);
        compressed_size += cmpSize;
        local_time += MPI_Wtime();

        // writefile
        cmpSize_set[id] = cmpSize;
        MPI_Allgather(MPI_IN_PLACE, 1, MPI_UNSIGNED_LONG, &cmpSize_set[0], 1, MPI_UNSIGNED_LONG, comm_val);
        auto mpi_offsets = compute_offsets(cmpSize_set, cmpSize_set.size());

        std::string compressed_file = compressed_file_prefix + varnames[i] + ".sz3_step" + std::to_string(timestep) + "_" + std::to_string(n_proc) + "_" + std::to_string(eb_rel);
        
        MPI_File fh;
        MPI_File_open(comm_val, compressed_file.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
        if(!id) MPI_File_write_at(fh, 0, cmpSize_set.data(), cmpSize_set.size(), MPI_UNSIGNED_LONG, MPI_STATUS_IGNORE);
        MPI_File_write_at(fh, mpi_offsets[id] + n_proc*sizeof(MPI_UNSIGNED_LONG), cmpData, cmpSize, MPI_CHAR, MPI_STATUS_IGNORE);
        MPI_File_close(&fh);
        free(cmpData);
    }
    double max_local_time = 0;
    MPI_Reduce(&local_time, &max_local_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm_val);

    size_t all_compressed_size = 0;
    MPI_Reduce(&compressed_size, &all_compressed_size, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, comm_val);

    if(!id){
        double aggregated_cr = n_vars*num_elements*sizeof(T) *1.0 / all_compressed_size;
        printf("aggregated_compression_ratio = %f\n", aggregated_cr);
        printf("compression_time = %f\n", max_local_time);
        fflush(stdout);
    }

}

template<class T>
void par_compress_sz3ori(const std::vector<std::string> varnames, size_t n_vars, int d, int maxEdges, int timestep, MPI_Comm comm_val, double eb_rel, int verb){
    int n_proc, id;
    MPI_Comm_size(comm_val, &n_proc);
    MPI_Comm_rank(comm_val, &id);
    
    // read partition
    std::string nparts_str = std::to_string(n_proc);
    std::string part_file = part_prefix + nparts_str + ".part";
    std::string part_map_file = part_prefix + nparts_str + ".part_map";
    std::string num_part_nodes_file = part_prefix + nparts_str + ".num_part_nodes";
    size_t num = 0;
    auto part = readfile<idx_t>(part_file.c_str(), num);
    auto part_map = readfile<uint32_t>(part_map_file.c_str(), num);
    auto num_part_nodes = readfile<int32_t>(num_part_nodes_file.c_str(), num);
    auto vtxdist = compute_offsets(num_part_nodes, n_proc);

    size_t num_elements = part.size();

    std::string data_file_prefix = data_file_dir + "step_" + std::to_string(timestep) + "/";
    std::string compressed_file_prefix = compressed_file_dir_sz3ori;
    par_compress_sz3ori<T>(timestep, varnames, data_file_prefix, compressed_file_prefix, n_vars, num_elements, eb_rel, part_map, num_part_nodes, vtxdist, comm_val, verb);
}


/* sz3 decompression without data reordering */
template<class T>
void par_decompress_sz3ori(int timestep, const std::vector<std::string> varnames, const std::string data_file_prefix, const std::string compressed_file_prefix, size_t n_vars, size_t num_elements,
                            double eb_rel, const std::vector<uint32_t>& part_map, const std::vector<int32_t>& num_part_nodes, const std::vector<int>& vtxdist,
                                MPI_Comm comm_val, int verb){
    int n_proc, id;
    MPI_Comm_size(comm_val, &n_proc);
    MPI_Comm_rank(comm_val, &id);
    
    MPI_Datatype mpi_dtype = get_mpi_dtype<T>();

    double decomp_time = 0;
    double time1;
    size_t local_size = num_part_nodes[id];
    
    for(int i=0; i<n_vars; i++){
        std::vector<size_t> cmpSize_set(n_proc);
        MPI_File fh_read;
        std::string compressed_file = compressed_file_prefix + varnames[i] + ".sz3_step" + std::to_string(timestep) + "_" + std::to_string(n_proc) + "_" + std::to_string(eb_rel);
        MPI_File_open(comm_val, compressed_file.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh_read);
        if(!id) MPI_File_read_at(fh_read, 0, cmpSize_set.data(), n_proc, MPI_UNSIGNED_LONG, MPI_STATUS_IGNORE);
        MPI_Bcast(cmpSize_set.data(), n_proc, MPI_UNSIGNED_LONG, 0, comm_val);
        auto mpi_offsets = compute_offsets(cmpSize_set, cmpSize_set.size());
        size_t cmpSize = cmpSize_set[id];
        char * input = (char *) malloc(cmpSize*sizeof(char));
        MPI_File_read_at(fh_read, mpi_offsets[id] + n_proc*sizeof(MPI_UNSIGNED_LONG), input, cmpSize, MPI_CHAR, MPI_STATUS_IGNORE);

        MPI_Barrier(comm_val);
        time1 = -MPI_Wtime();
        auto decData = new T[local_size];
        SZ3_decompress(input, cmpSize, decData);
        time1 += MPI_Wtime();
        decomp_time += time1;

        std::vector<T> var_dec_data(num_elements);
        std::vector<T> var_local_dec_data(decData, decData+local_size);
        MPI_Gatherv(var_local_dec_data.data(), local_size, mpi_dtype, &var_dec_data[0], &num_part_nodes[0], &vtxdist[0], mpi_dtype, 0, comm_val);

        if(!id){
            size_t num = 0;
            std::string data_file = data_file_prefix + varnames[i];
            std::vector<T> data_ori = readfile<T>(data_file.c_str(), num);
            std::vector<T> dec_data_original_order;
            dec_data_original_order = reorder_back(var_dec_data, part_map, num_elements, 1);
            if(verb){
                std::cout << i << ": " << std::endl;
                print_statistics(data_ori.data(), dec_data_original_order.data(), num_elements);
            }
            writefile((data_file+".para.sz3.out").c_str(), dec_data_original_order.data(), num_elements);
        }
        delete[] decData;
        free(input);
    }

    double max_decomp_time = 0;
    MPI_Reduce(&decomp_time, &max_decomp_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm_val);
    if(!id){
        printf("decomprression_time = %f\n", max_decomp_time);
        fflush(stdout);
    }
}

template<class T>
void par_decompress_sz3ori(const std::vector<std::string> varnames, size_t n_vars, int d, int maxEdges, int timestep, MPI_Comm comm_val, double eb_rel, int verb){
    int n_proc, id;
    MPI_Comm_size(comm_val, &n_proc);
    MPI_Comm_rank(comm_val, &id);
    
    // read partition
    std::string nparts_str = std::to_string(n_proc);
    std::string part_file = part_prefix + nparts_str + ".part";
    std::string part_map_file = part_prefix + nparts_str + ".part_map";
    std::string num_part_nodes_file = part_prefix + nparts_str + ".num_part_nodes";
    size_t num = 0;
    auto part = readfile<idx_t>(part_file.c_str(), num);
    auto part_map = readfile<uint32_t>(part_map_file.c_str(), num);
    auto num_part_nodes = readfile<int32_t>(num_part_nodes_file.c_str(), num);
    auto vtxdist = compute_offsets(num_part_nodes, n_proc);

    size_t num_elements = part.size();
    std::string data_file_prefix = data_file_dir + "step_" + std::to_string(timestep) + "/";
    std::string compressed_file_prefix = compressed_file_dir_sz3ori;
    par_decompress_sz3ori<T>(timestep, varnames, data_file_prefix, compressed_file_prefix, n_vars, num_elements, eb_rel, part_map, num_part_nodes, vtxdist, comm_val, verb);
}


/* sz3 compression using PDFS data reordering */
template<class T>
void par_compress_sz3rod(int timestep, const std::vector<std::string> varnames, const std::string data_file_prefix, const std::string compressed_file_prefix, size_t n_vars,
                            size_t num_elements, double eb_rel, const std::vector<int32_t>& index_map, const std::vector<uint32_t>& part_map,
                                const std::vector<int32_t>& num_part_nodes, const std::vector<int>& vtxdist, MPI_Comm comm_val, int verb){
    int n_proc, id;
    MPI_Comm_size(comm_val, &n_proc);
    MPI_Comm_rank(comm_val, &id);

    MPI_Datatype mpi_dtype = get_mpi_dtype<T>();

    size_t compressed_size = 0;
    size_t local_size = num_part_nodes[id];
    size_t num = 0;

    double local_time = 0;
    for(int i=0; i<n_vars; i++){
        std::vector<size_t> cmpSize_set(n_proc);

        std::vector<T> stats(3);
        std::vector<T> local_data(local_size), local_data1(local_size);
        std::string data_file = data_file_prefix + varnames[i];
        std::vector<T> data, data1;
        if(!id){
            data.resize(num_elements);
            data = readfile<T>(data_file.c_str(), num);
            data1 = reorder(data, part_map, num_elements, 1);
            stats[0] = *std::min_element(data.begin(), data.end());
            stats[1] = *std::max_element(data.begin(), data.end());
            stats[2] = stats[1] - stats[0];
        }
        MPI_Bcast(&stats[0], 3, mpi_dtype, 0, comm_val);
        MPI_Scatterv(&data1[0], &num_part_nodes[0], &vtxdist[0], mpi_dtype, &local_data[0], local_size, mpi_dtype, 0, comm_val);
        
        MPI_Barrier(comm_val);
        local_time -= MPI_Wtime();
        double eb = eb_rel * stats[2];
        size_t cmpSize = 0;
        char * cmpData = SZ3_compress_1D(local_size, local_data1.data(), eb, cmpSize);
        local_time += MPI_Wtime();
        compressed_size += cmpSize;

        // writefile
        cmpSize_set[id] = cmpSize;
        MPI_Allgather(MPI_IN_PLACE, 1, MPI_UNSIGNED_LONG, &cmpSize_set[0], 1, MPI_UNSIGNED_LONG, comm_val);
        auto mpi_offsets = compute_offsets(cmpSize_set, cmpSize_set.size());

        std::string compressed_file = compressed_file_prefix + varnames[i] + ".sz3_step" + std::to_string(timestep) + "_" + std::to_string(n_proc) + "_" + std::to_string(eb_rel);
        
        MPI_File fh;
        MPI_File_open(comm_val, compressed_file.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
        if(!id) MPI_File_write_at(fh, 0, cmpSize_set.data(), cmpSize_set.size(), MPI_UNSIGNED_LONG, MPI_STATUS_IGNORE);
        MPI_File_write_at(fh, mpi_offsets[id] + n_proc*sizeof(MPI_UNSIGNED_LONG), cmpData, cmpSize, MPI_CHAR, MPI_STATUS_IGNORE);
        MPI_File_close(&fh);
        free(cmpData);
    }

    double max_local_time = 0;
    MPI_Reduce(&local_time, &max_local_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm_val);

    size_t all_compressed_size = 0;
    MPI_Reduce(&compressed_size, &all_compressed_size, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, comm_val);

    if(!id){
        double aggregated_cr = n_vars*num_elements*sizeof(T) *1.0 / all_compressed_size;
        printf("aggregated_compression_ratio = %f\n", aggregated_cr);
        printf("compression_time = %f\n", max_local_time);
        fflush(stdout);
    }

}

template<class T>
void par_compress_sz3rod(const std::vector<std::string> varnames, size_t n_vars, int d, int maxEdges, int timestep, MPI_Comm comm_val, double eb_rel, int verb){
    int n_proc, id;
    MPI_Comm_size(comm_val, &n_proc);
    MPI_Comm_rank(comm_val, &id);
    
    // read partition
    std::string nparts_str = std::to_string(n_proc);
    std::string part_file = part_prefix + nparts_str + ".part";
    std::string part_map_file = part_prefix + nparts_str + ".part_map";
    std::string num_part_nodes_file = part_prefix + nparts_str + ".num_part_nodes";
    size_t num = 0;
    auto conn = readfile<int>(conn_file.c_str(), num);
    auto part = readfile<idx_t>(part_file.c_str(), num);
    auto part_map = readfile<uint32_t>(part_map_file.c_str(), num);
    auto num_part_nodes = readfile<int32_t>(num_part_nodes_file.c_str(), num);
    auto vtxdist = compute_offsets(num_part_nodes, n_proc);

    size_t num_elements = part.size();

    double prep_time = 0;
    MPI_Barrier(comm_val);
    prep_time = -MPI_Wtime();

    auto local_adj_list = MPAS_generate_local_adjacent_list(id, num_elements, maxEdges, conn, part, part_map, vtxdist[id], num_part_nodes[id]);
    auto index_map = generate_reorder_index_map(num_part_nodes[id], local_adj_list, 1);

    MPI_Barrier(comm_val);
    prep_time += MPI_Wtime();

    if(!id){
        printf("preprocessing_time = %f\n", prep_time);
        fflush(stdout);
    }

    local_adj_list = std::vector<std::set<int32_t>>();
    conn = std::vector<int>();
    
    std::string data_file_prefix = data_file_dir + "step_" + std::to_string(timestep) + "/";
    std::string compressed_file_prefix = compressed_file_dir_sz3rod;

    par_compress_sz3rod<T>(timestep, varnames, data_file_prefix, compressed_file_prefix, n_vars, num_elements, eb_rel, index_map, part_map, num_part_nodes, vtxdist, comm_val, verb);
}

/* sz3 decompression using PDFS data reordering */
template<class T>
void par_decompress_sz3rod(int timestep, const std::vector<std::string> varnames, const std::string data_file_prefix, const std::string compressed_file_prefix, size_t n_vars, size_t num_elements,
                            double eb_rel, const std::vector<int32_t>& index_map, const std::vector<uint32_t>& part_map, const std::vector<int32_t>& num_part_nodes, const std::vector<int>& vtxdist,
                                MPI_Comm comm_val, int verb){
    int n_proc, id;
    MPI_Comm_size(comm_val, &n_proc);
    MPI_Comm_rank(comm_val, &id);
    
    MPI_Datatype mpi_dtype = get_mpi_dtype<T>();

    double decomp_time = 0;
    double time1;
    size_t local_size = num_part_nodes[id];
    
    for(int i=0; i<n_vars; i++){
        std::vector<size_t> cmpSize_set(n_proc);
        MPI_File fh_read;
        std::string compressed_file = compressed_file_prefix + varnames[i] + ".sz3_step" + std::to_string(timestep) + "_" + std::to_string(n_proc) + "_" + std::to_string(eb_rel);
        MPI_File_open(comm_val, compressed_file.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh_read);
        if(!id) MPI_File_read_at(fh_read, 0, cmpSize_set.data(), n_proc, MPI_UNSIGNED_LONG, MPI_STATUS_IGNORE);
        MPI_Bcast(cmpSize_set.data(), n_proc, MPI_UNSIGNED_LONG, 0, comm_val);
        auto mpi_offsets = compute_offsets(cmpSize_set, cmpSize_set.size());
        size_t cmpSize = cmpSize_set[id];
        char * input = (char *) malloc(cmpSize*sizeof(char));
        MPI_File_read_at(fh_read, mpi_offsets[id] + n_proc*sizeof(MPI_UNSIGNED_LONG), input, cmpSize, MPI_CHAR, MPI_STATUS_IGNORE);

        MPI_Barrier(comm_val);
        time1 = -MPI_Wtime();
        auto decData = new T[local_size];
        SZ3_decompress(input, cmpSize, decData);
        time1 += MPI_Wtime();
        decomp_time += time1;

        std::vector<T> var_dec_data(num_elements);
        std::vector<T> var_local_dec_data(decData, decData+local_size);
        std::vector<T> interm_data = reorder_back(var_local_dec_data, index_map, local_size, 1);
        MPI_Gatherv(interm_data.data(), local_size, mpi_dtype, &var_dec_data[0], &num_part_nodes[0], &vtxdist[0], mpi_dtype, 0, comm_val);

        if(!id){
            size_t num = 0;
            std::string data_file = data_file_prefix + varnames[i];
            std::vector<T> data_ori = readfile<T>(data_file.c_str(), num);
            std::vector<T> dec_data_original_order;
            dec_data_original_order = reorder_back(var_dec_data, part_map, num_elements, 1);
            // if(verb){
            //     std::cout << i << ": " << std::endl;
            //     print_statistics(data_ori.data(), dec_data_original_order.data(), num_elements);
            // }
            writefile((data_file+".para.sz3.out").c_str(), dec_data_original_order.data(), num_elements);
        }
        delete[] decData;
        free(input);
    }

    double max_decomp_time = 0;
    MPI_Reduce(&decomp_time, &max_decomp_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm_val);
    if(!id){
        printf("decomprression_time = %f\n", max_decomp_time);
        fflush(stdout);
    }
}

template<class T>
void par_decompress_sz3rod(const std::vector<std::string> varnames, size_t n_vars, int d, int maxEdges, int timestep, MPI_Comm comm_val, double eb_rel, int verb){
    int n_proc, id;
    MPI_Comm_size(comm_val, &n_proc);
    MPI_Comm_rank(comm_val, &id);
    
    // read partition
    std::string nparts_str = std::to_string(n_proc);
    std::string part_file = part_prefix + nparts_str + ".part";
    std::string part_map_file = part_prefix + nparts_str + ".part_map";
    std::string num_part_nodes_file = part_prefix + nparts_str + ".num_part_nodes";
    size_t num = 0;
    auto conn = readfile<int>(conn_file.c_str(), num);
    auto part = readfile<idx_t>(part_file.c_str(), num);
    auto part_map = readfile<uint32_t>(part_map_file.c_str(), num);
    auto num_part_nodes = readfile<int32_t>(num_part_nodes_file.c_str(), num);
    auto vtxdist = compute_offsets(num_part_nodes, n_proc);

    size_t num_elements = part.size();

    double prep_time = 0;

    MPI_Barrier(comm_val);
    prep_time -= MPI_Wtime();

    auto local_adj_list = MPAS_generate_local_adjacent_list(id, num_elements, maxEdges, conn, part, part_map, vtxdist[id], num_part_nodes[id]);
    auto index_map = generate_reorder_index_map(num_part_nodes[id], local_adj_list, 1);

    MPI_Barrier(comm_val);
    prep_time += MPI_Wtime();

    local_adj_list = std::vector<std::set<int32_t>>();
    conn = std::vector<int>();

    std::string data_file_prefix = data_file_dir + "step_" + std::to_string(timestep) + "/";
    std::string compressed_file_prefix = compressed_file_dir_sz3rod;
    par_decompress_sz3rod<T>(timestep, varnames, data_file_prefix, compressed_file_prefix, n_vars, num_elements, eb_rel, index_map, part_map, num_part_nodes, vtxdist, comm_val, verb);

    if(!id){
        printf("preprocessing_time = %f\n", prep_time);
        fflush(stdout);
    }
}

/* zfp compression without data reordering */
template<class T>
void par_compress_zfpori(int timestep, const std::vector<std::string> varnames, const std::string data_file_prefix, const std::string compressed_file_prefix, size_t n_vars, size_t num_elements, double eb_rel,
                        const std::vector<uint32_t>& part_map, const std::vector<int32_t>& num_part_nodes, const std::vector<int>& vtxdist, MPI_Comm comm_val, int verb){
    int n_proc, id;
    MPI_Comm_size(comm_val, &n_proc);
    MPI_Comm_rank(comm_val, &id);

    MPI_Datatype mpi_dtype = get_mpi_dtype<T>();

    size_t compressed_size = 0;
    size_t local_size = num_part_nodes[id];
    size_t num = 0;

    double local_time = 0;
    for(int i=0; i<n_vars; i++){
        std::vector<size_t> cmpSize_set(n_proc);

        std::vector<T> stats(3);
        std::vector<T> local_data(local_size);
        std::string data_file = data_file_prefix + varnames[i];
        std::vector<T> data, data1;
        if(!id){
            data.resize(num_elements);
            data = readfile<T>(data_file.c_str(), num);
            data1 = reorder(data, part_map, num_elements, 1);
            stats[0] = *std::min_element(data.begin(), data.end());
            stats[1] = *std::max_element(data.begin(), data.end());
            stats[2] = stats[1] - stats[0];
        }
        MPI_Bcast(&stats[0], 3, mpi_dtype, 0, comm_val);
        MPI_Scatterv(&data1[0], &num_part_nodes[0], &vtxdist[0], mpi_dtype, &local_data[0], local_size, mpi_dtype, 0, comm_val);
        
        double eb = eb_rel * stats[2];
        size_t cmpSize = 0;

        MPI_Barrier(comm_val);
        local_time -= MPI_Wtime();
        unsigned char * cmpData = zfp_compress_1D(local_data.data(), eb, local_size, cmpSize);
        compressed_size += cmpSize;
        local_time += MPI_Wtime();

        // writefile
        cmpSize_set[id] = cmpSize;
        MPI_Allgather(MPI_IN_PLACE, 1, MPI_UNSIGNED_LONG, &cmpSize_set[0], 1, MPI_UNSIGNED_LONG, comm_val);
        auto mpi_offsets = compute_offsets(cmpSize_set, cmpSize_set.size());

        std::string compressed_file = compressed_file_prefix + varnames[i] + ".zfp_step" + std::to_string(timestep) + "_" + std::to_string(n_proc) + "_" + std::to_string(eb_rel);
        
        MPI_File fh;
        MPI_File_open(comm_val, compressed_file.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
        if(!id) MPI_File_write_at(fh, 0, cmpSize_set.data(), cmpSize_set.size(), MPI_UNSIGNED_LONG, MPI_STATUS_IGNORE);
        MPI_File_write_at(fh, mpi_offsets[id] + n_proc*sizeof(MPI_UNSIGNED_LONG), cmpData, cmpSize, MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);
        MPI_File_close(&fh);
    }
    double max_local_time = 0;
    MPI_Reduce(&local_time, &max_local_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm_val);

    size_t all_compressed_size = 0;
    MPI_Reduce(&compressed_size, &all_compressed_size, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, comm_val);

    if(!id){
        double aggregated_cr = n_vars*num_elements*sizeof(T) *1.0 / all_compressed_size;
        printf("aggregated_compression_ratio = %f\n", aggregated_cr);
        printf("compression_time = %f\n", max_local_time);
        fflush(stdout);
    }

}

template<class T>
void par_compress_zfpori(const std::vector<std::string> varnames, size_t n_vars, int d, int maxEdges, int timestep,
                        MPI_Comm comm_val, double eb_rel, int verb){
    int n_proc, id;
    MPI_Comm_size(comm_val, &n_proc);
    MPI_Comm_rank(comm_val, &id);

    // read partition
    std::string nparts_str = std::to_string(n_proc);
    std::string part_file = part_prefix + nparts_str + ".part";
    std::string part_map_file = part_prefix + nparts_str + ".part_map";
    std::string num_part_nodes_file = part_prefix + nparts_str + ".num_part_nodes";
    size_t num = 0;
    auto part = readfile<idx_t>(part_file.c_str(), num);
    auto part_map = readfile<uint32_t>(part_map_file.c_str(), num);
    auto num_part_nodes = readfile<int32_t>(num_part_nodes_file.c_str(), num);
    auto vtxdist = compute_offsets(num_part_nodes, n_proc);

    size_t num_elements = part.size();

    std::string data_file_prefix = data_file_dir + "step_" + std::to_string(timestep) + "/";
    std::string compressed_file_prefix = compressed_file_dir_zfpori;
    par_compress_zfpori<T>(timestep, varnames, data_file_prefix, compressed_file_prefix, n_vars, num_elements, eb_rel, part_map, num_part_nodes, vtxdist, comm_val, verb);
}

/* zfp decompression without data reordering */
template<class T>
void par_decompress_zfpori(int timestep, const std::vector<std::string> varnames, const std::string data_file_prefix, const std::string compressed_file_prefix, size_t n_vars, size_t num_elements,
                        double eb_rel, const std::vector<uint32_t>& part_map, const std::vector<int32_t>& num_part_nodes, const std::vector<int>& vtxdist,
                            MPI_Comm comm_val, int verb){
    int n_proc, id;
    MPI_Comm_size(comm_val, &n_proc);
    MPI_Comm_rank(comm_val, &id);
    
    MPI_Datatype mpi_dtype = get_mpi_dtype<T>();

    double decomp_time = 0;
    double time1;
    size_t local_size = num_part_nodes[id];
    
    for(int i=0; i<n_vars; i++){
        std::vector<T> stats(3);
        std::vector<T> data_ori;
        std::string data_file = data_file_prefix + varnames[i];
        if(!id){
            size_t num = 0;
            data_ori = readfile<T>(data_file.c_str(), num);
            auto min_max = std::minmax_element(data_ori.begin(), data_ori.end());
            stats[0] = *min_max.first;
            stats[1] = *min_max.second;
            stats[2] = stats[1] - stats[0];
        }
        MPI_Bcast(stats.data(), 3, mpi_dtype, 0, comm_val);

        std::vector<size_t> cmpSize_set(n_proc);
        MPI_File fh_read;
        std::string compressed_file = compressed_file_prefix + varnames[i] + ".zfp_step" + std::to_string(timestep) + "_" + std::to_string(n_proc) + "_" + std::to_string(eb_rel);
        MPI_File_open(comm_val, compressed_file.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh_read);
        if(!id) MPI_File_read_at(fh_read, 0, cmpSize_set.data(), n_proc, MPI_UNSIGNED_LONG, MPI_STATUS_IGNORE);
        MPI_Bcast(cmpSize_set.data(), n_proc, MPI_UNSIGNED_LONG, 0, comm_val);
        auto mpi_offsets = compute_offsets(cmpSize_set, cmpSize_set.size());
        size_t cmpSize = cmpSize_set[id];
        unsigned char * input = (unsigned char *) malloc(cmpSize*sizeof(unsigned char));
        MPI_File_read_at(fh_read, mpi_offsets[id] + n_proc*sizeof(MPI_UNSIGNED_LONG), input, cmpSize, MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);

        double eb = eb_rel*stats[2];

        MPI_Barrier(comm_val);
        time1 = -MPI_Wtime();
        float * decData = zfp_decompress_1D(input, eb, local_size*2, local_size);
        time1 += MPI_Wtime();
        decomp_time += time1;

        std::vector<T> var_dec_data(num_elements);
        std::vector<T> var_local_dec_data(decData, decData+local_size);
        MPI_Gatherv(var_local_dec_data.data(), local_size, mpi_dtype, &var_dec_data[0], &num_part_nodes[0], &vtxdist[0], mpi_dtype, 0, comm_val);

        if(!id){
            std::vector<T> dec_data_original_order;
            dec_data_original_order = reorder_back(var_dec_data, part_map, num_elements, 1);
            if(verb){
                std::cout << i << ": " << std::endl;
                print_statistics(data_ori.data(), dec_data_original_order.data(), num_elements);
            }
            writefile((data_file+".para.zfp.out").c_str(), dec_data_original_order.data(), num_elements);
        }
        free(decData);
        free(input);
    }

    double max_decomp_time = 0;
    MPI_Reduce(&decomp_time, &max_decomp_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm_val);
    if(!id){
        printf("decomprression_time = %f\n", max_decomp_time);
        fflush(stdout);
    }
}

template<class T>
void par_decompress_zfpori(const std::vector<std::string> varnames, size_t n_vars, int d, int maxEdges, int timestep, MPI_Comm comm_val, double eb_rel, int verb){
    int n_proc, id;
    MPI_Comm_size(comm_val, &n_proc);
    MPI_Comm_rank(comm_val, &id);

    // read partition
    std::string nparts_str = std::to_string(n_proc);
    std::string part_file = part_prefix + nparts_str + ".part";
    std::string part_map_file = part_prefix + nparts_str + ".part_map";
    std::string num_part_nodes_file = part_prefix + nparts_str + ".num_part_nodes";
    size_t num = 0;
    auto part = readfile<idx_t>(part_file.c_str(), num);
    auto part_map = readfile<uint32_t>(part_map_file.c_str(), num);
    auto num_part_nodes = readfile<int32_t>(num_part_nodes_file.c_str(), num);
    auto vtxdist = compute_offsets(num_part_nodes, n_proc);

    size_t num_elements = part.size();

    std::string data_file_prefix = data_file_dir + "step_" + std::to_string(timestep) + "/";
    std::string compressed_file_prefix = compressed_file_dir_zfpori;
    par_decompress_zfpori<T>(timestep, varnames, data_file_prefix, compressed_file_prefix, n_vars, num_elements, eb_rel, part_map, num_part_nodes, vtxdist, comm_val, verb);
}


/* zfp compression using PDFS data reordering */
template<class T>
void par_compress_zfprod(int timestep, const std::vector<std::string> varnames, const std::string data_file_prefix, const std::string compressed_file_prefix, size_t n_vars, size_t num_elements, double eb_rel, const std::vector<int32_t>& index_map,
                        const std::vector<uint32_t>& part_map, const std::vector<int32_t>& num_part_nodes, const std::vector<int>& vtxdist, MPI_Comm comm_val, int verb){
    int n_proc, id;
    MPI_Comm_size(comm_val, &n_proc);
    MPI_Comm_rank(comm_val, &id);

    MPI_Datatype mpi_dtype = get_mpi_dtype<T>();

    size_t compressed_size = 0;
    size_t local_size = num_part_nodes[id];
    size_t num = 0;

    double local_time = 0;
    for(int i=0; i<n_vars; i++){
        std::vector<size_t> cmpSize_set(n_proc);

        std::vector<T> stats(3);
        std::vector<T> local_data(local_size), local_data1(local_size);
        std::string data_file = data_file_prefix + varnames[i];
        std::vector<T> data, data1;
        if(!id){
            data.resize(num_elements);
            data = readfile<T>(data_file.c_str(), num);
            data1 = reorder(data, part_map, num_elements, 1);
            stats[0] = *std::min_element(data.begin(), data.end());
            stats[1] = *std::max_element(data.begin(), data.end());
            stats[2] = stats[1] - stats[0];
        }
        MPI_Bcast(&stats[0], 3, mpi_dtype, 0, comm_val);
        MPI_Scatterv(&data1[0], &num_part_nodes[0], &vtxdist[0], mpi_dtype, &local_data[0], local_size, mpi_dtype, 0, comm_val);

        double eb = eb_rel * stats[2];  
        size_t cmpSize = 0;

        MPI_Barrier(comm_val);
        local_time -= MPI_Wtime();
        local_data1 = reorder(local_data, index_map, local_size, 1);
        unsigned char * cmpData = zfp_compress_1D(local_data1.data(), eb, local_size, cmpSize);
        compressed_size += cmpSize;
        local_time += MPI_Wtime();

        // writefile
        cmpSize_set[id] = cmpSize;
        MPI_Allgather(MPI_IN_PLACE, 1, MPI_UNSIGNED_LONG, &cmpSize_set[0], 1, MPI_UNSIGNED_LONG, comm_val);
        auto mpi_offsets = compute_offsets(cmpSize_set, cmpSize_set.size());

        std::string compressed_file = compressed_file_prefix + varnames[i] + ".zfp_step" + std::to_string(timestep) + "_" + std::to_string(n_proc) + "_" + std::to_string(eb_rel);
        
        MPI_File fh;
        MPI_File_open(comm_val, compressed_file.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
        if(!id) MPI_File_write_at(fh, 0, cmpSize_set.data(), cmpSize_set.size(), MPI_UNSIGNED_LONG, MPI_STATUS_IGNORE);
        MPI_File_write_at(fh, mpi_offsets[id] + n_proc*sizeof(MPI_UNSIGNED_LONG), cmpData, cmpSize, MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);
        MPI_File_close(&fh);
    }

    double max_local_time = 0;
    MPI_Reduce(&local_time, &max_local_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm_val);

    size_t all_compressed_size = 0;
    MPI_Reduce(&compressed_size, &all_compressed_size, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, comm_val);

    if(!id){
        double aggregated_cr = n_vars*num_elements*sizeof(T) *1.0 / all_compressed_size;
        printf("aggregated_compression_ratio = %f\n", aggregated_cr);
        printf("compression_time = %f\n", max_local_time);
        fflush(stdout);
    }

}

template<class T>
void par_compress_zfprod(const std::vector<std::string> varnames, size_t n_vars, int d, int maxEdges, int timestep,
                        MPI_Comm comm_val, double eb_rel, int verb){
    int n_proc, id;
    MPI_Comm_size(comm_val, &n_proc);
    MPI_Comm_rank(comm_val, &id);
    
    // read partition
    std::string nparts_str = std::to_string(n_proc);
    std::string part_file = part_prefix + nparts_str + ".part";
    std::string part_map_file = part_prefix + nparts_str + ".part_map";
    std::string num_part_nodes_file = part_prefix + nparts_str + ".num_part_nodes";
    size_t num = 0;
    auto conn = readfile<int>(conn_file.c_str(), num);
    auto part = readfile<idx_t>(part_file.c_str(), num);
    auto part_map = readfile<uint32_t>(part_map_file.c_str(), num);
    auto num_part_nodes = readfile<int32_t>(num_part_nodes_file.c_str(), num);
    auto vtxdist = compute_offsets(num_part_nodes, n_proc);

    size_t num_elements = part.size();

    double prep_time = 0;
    MPI_Barrier(comm_val);
    prep_time = -MPI_Wtime();

    auto local_adj_list = MPAS_generate_local_adjacent_list(id, num_elements, maxEdges, conn, part, part_map, vtxdist[id], num_part_nodes[id]);
    auto index_map = generate_reorder_index_map(num_part_nodes[id], local_adj_list, 1);

    MPI_Barrier(comm_val);
    prep_time += MPI_Wtime();

    if(!id){
        printf("preprocessing_time = %f\n", prep_time);
        fflush(stdout);
    }

    local_adj_list = std::vector<std::set<int32_t>>();
    conn = std::vector<int>();

    std::string data_file_prefix = data_file_dir + "step_" + std::to_string(timestep) + "/";
    std::string compressed_file_prefix = compressed_file_dir_zfprod;
    par_compress_zfprod<T>(timestep, varnames, data_file_prefix, compressed_file_prefix, n_vars, num_elements, eb_rel, index_map, part_map, num_part_nodes, vtxdist, comm_val, verb);
}

/* zfp decompression using PDFS data reordering */
template<class T>
void par_decompress_zfprod(int timestep, const std::vector<std::string> varnames, const std::string data_file_prefix, const std::string compressed_file_prefix, size_t n_vars, size_t num_elements,
                        double eb_rel, const std::vector<int32_t>& index_map, const std::vector<uint32_t>& part_map, const std::vector<int32_t>& num_part_nodes, const std::vector<int>& vtxdist,
                            MPI_Comm comm_val, int verb){
    int n_proc, id;
    MPI_Comm_size(comm_val, &n_proc);
    MPI_Comm_rank(comm_val, &id);
    
    MPI_Datatype mpi_dtype = get_mpi_dtype<T>();

    double decomp_time = 0;
    double time1;
    size_t local_size = num_part_nodes[id];
    
    for(int i=0; i<n_vars; i++){
        std::vector<T> stats(3);
        std::vector<T> data_ori;
        std::string data_file = data_file_prefix + varnames[i];
        if(!id){
            size_t num = 0;
            data_ori = readfile<T>(data_file.c_str(), num);
            auto min_max = std::minmax_element(data_ori.begin(), data_ori.end());
            stats[0] = *min_max.first;
            stats[1] = *min_max.second;
            stats[2] = stats[1] - stats[0];
        }
        MPI_Bcast(stats.data(), 3, mpi_dtype, 0, comm_val);

        std::vector<size_t> cmpSize_set(n_proc);
        MPI_File fh_read;
        std::string compressed_file = compressed_file_prefix + varnames[i] + ".zfp_step" + std::to_string(timestep) + "_" + std::to_string(n_proc) + "_" + std::to_string(eb_rel);
        MPI_File_open(comm_val, compressed_file.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh_read);
        if(!id) MPI_File_read_at(fh_read, 0, cmpSize_set.data(), n_proc, MPI_UNSIGNED_LONG, MPI_STATUS_IGNORE);
        MPI_Bcast(cmpSize_set.data(), n_proc, MPI_UNSIGNED_LONG, 0, comm_val);
        auto mpi_offsets = compute_offsets(cmpSize_set, cmpSize_set.size());
        size_t cmpSize = cmpSize_set[id];
        unsigned char * input = (unsigned char *) malloc(cmpSize*sizeof(unsigned char));
        MPI_File_read_at(fh_read, mpi_offsets[id] + n_proc*sizeof(MPI_UNSIGNED_LONG), input, cmpSize, MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);

        double eb = eb_rel*stats[2];

        MPI_Barrier(comm_val);
        time1 = -MPI_Wtime();
        float * decData = zfp_decompress_1D(input, eb, local_size*2, local_size);
        time1 += MPI_Wtime();
        decomp_time += time1;

        std::vector<T> var_dec_data(num_elements);
        std::vector<T> var_local_dec_data(decData, decData+local_size);
        std::vector<T> interm_data = reorder_back(var_local_dec_data, index_map, local_size, 1);
        MPI_Gatherv(interm_data.data(), local_size, mpi_dtype, &var_dec_data[0], &num_part_nodes[0], &vtxdist[0], mpi_dtype, 0, comm_val);

        if(!id){
            std::vector<T> dec_data_original_order;
            dec_data_original_order = reorder_back(var_dec_data, part_map, num_elements, 1);
            if(verb){
                std::cout << i << ": " << std::endl;
                print_statistics(data_ori.data(), dec_data_original_order.data(), num_elements);
            }
            writefile((data_file+".para.zfp.out").c_str(), dec_data_original_order.data(), num_elements);
        }
        free(decData);
        free(input);
    }

    double max_decomp_time = 0;
    MPI_Reduce(&decomp_time, &max_decomp_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm_val);
    if(!id){
        printf("decomprression_time = %f\n", max_decomp_time);
        fflush(stdout);
    }
}

template<class T>
void par_decompress_zfprod(const std::vector<std::string> varnames, size_t n_vars, int d, int maxEdges, int timestep, MPI_Comm comm_val, double eb_rel, int verb){
    int n_proc, id;
    MPI_Comm_size(comm_val, &n_proc);
    MPI_Comm_rank(comm_val, &id);
    
    // read partition
    std::string nparts_str = std::to_string(n_proc);
    std::string part_file = part_prefix + nparts_str + ".part";
    std::string part_map_file = part_prefix + nparts_str + ".part_map";
    std::string num_part_nodes_file = part_prefix + nparts_str + ".num_part_nodes";
    size_t num = 0;
    auto conn = readfile<int>(conn_file.c_str(), num);
    auto part = readfile<idx_t>(part_file.c_str(), num);
    auto part_map = readfile<uint32_t>(part_map_file.c_str(), num);
    auto num_part_nodes = readfile<int32_t>(num_part_nodes_file.c_str(), num);
    auto vtxdist = compute_offsets(num_part_nodes, n_proc);

    size_t num_elements = part.size();

    double prep_time = 0;

    MPI_Barrier(comm_val);
    prep_time -= MPI_Wtime();

    auto local_adj_list = MPAS_generate_local_adjacent_list(id, num_elements, maxEdges, conn, part, part_map, vtxdist[id], num_part_nodes[id]);
    auto index_map = generate_reorder_index_map(num_part_nodes[id], local_adj_list, 1);

    MPI_Barrier(comm_val);
    prep_time += MPI_Wtime();

    local_adj_list = std::vector<std::set<int32_t>>();
    conn = std::vector<int>();
    
    std::string data_file_prefix = data_file_dir + "step_" + std::to_string(timestep) + "/";
    std::string compressed_file_prefix = compressed_file_dir_zfprod;
    par_decompress_zfprod<T>(timestep, varnames, data_file_prefix, compressed_file_prefix, n_vars, num_elements, eb_rel, index_map, part_map, num_part_nodes, vtxdist, comm_val, verb);

    if(!id){
        printf("preprocessing_time = %f\n", prep_time);
        fflush(stdout);
    }
}

}
#endif