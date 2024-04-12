find_package(MPI REQUIRED)

add_executable(test_par_compress test_par_compress.cpp)
target_include_directories(test_par_compress PRIVATE ${ZFP_INCLUDE_DIR} ${MPI_INCLUDE_PATH})
target_link_libraries(test_par_compress ${PROJECT_NAME} ${ZFP_LIBRARY} MPI::MPI_CXX)

add_executable(test_par_decompress test_par_decompress.cpp)
target_include_directories(test_par_decompress PRIVATE ${ZFP_INCLUDE_DIR} ${MPI_INCLUDE_PATH})
target_link_libraries(test_par_decompress ${PROJECT_NAME} ${ZFP_LIBRARY} MPI::MPI_CXX)

add_executable(test_partition test_partition.cpp)
target_include_directories(test_partition PRIVATE ${ZFP_INCLUDE_DIR} ${MPI_INCLUDE_PATH})
target_link_libraries(test_partition ${PROJECT_NAME} ${ZFP_LIBRARY} MPI::MPI_CXX)