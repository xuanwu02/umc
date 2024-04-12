#include <mpi.h>
#include "par_compress_decompress.hpp"

using namespace UMC;

int main(int argc, char ** argv){

    MPI_Init(&argc, &argv);
    MPI_Barrier(MPI_COMM_WORLD);

    int n_proc, id;
    MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm comm_val = MPI_COMM_WORLD;

    using T = float;
    using T1 = double;

    int arg_pos = 1;
    int d = 3, maxEdges = 7;
    int method = atoi(argv[arg_pos++]);
    int reorder = atoi(argv[arg_pos++]);
    int timestep = atoi(argv[arg_pos++]);
    int verb = atoi(argv[arg_pos++]);
    int max_neighbors = atoi(argv[arg_pos++]);
    double eb_rel = atof(argv[arg_pos++]);

    std::vector<std::string> varnames = read_varnames();
    size_t n_vars = varnames.size();

    switch(method){
        case 0:
            if(!id){
                printf("Compress using IDW with PDFS reordering\n");
                fflush(stdout);
            }
            par_compress_idw<T, T1>(varnames, n_vars, d, maxEdges, timestep, max_neighbors, comm_val, eb_rel, verb);
        case 1:
            if(reorder){
                if(!id){
                    printf("Compress using SZ3 with PDFS reordering\n");
                    fflush(stdout);
                }
                par_compress_sz3rod<T>(varnames, n_vars, d, maxEdges, timestep, comm_val, eb_rel, verb);
            }
            else{
                if(!id){
                    printf("Compress using SZ3\n");
                    fflush(stdout);
                }
                par_compress_sz3ori<T>(varnames, n_vars, d, maxEdges, timestep, comm_val, eb_rel, verb);                
            }
        case 2:
            if(reorder){
                if(!id){
                    printf("Compress using ZFP with PDFS reordering\n");
                    fflush(stdout);
                }
                par_compress_zfprod<T>(varnames, n_vars, d, maxEdges, timestep, comm_val, eb_rel, verb);
            }
            else{
                if(!id){
                    printf("Compress using ZFP\n");
                    fflush(stdout);
                }
                par_compress_zfpori<T>(varnames, n_vars, d, maxEdges, timestep, comm_val, eb_rel, verb);                
            }
        default:
            break;
    }

    MPI_Finalize();
    return 0;
}