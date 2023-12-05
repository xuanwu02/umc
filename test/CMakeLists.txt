add_executable(test_compress test_compress.cpp)
target_link_libraries(test_compress ${PROJECT_NAME} ${METIS_LIBRARY} GSL::gsl SZ3::SZ3)

add_executable(test_decompress test_decompress.cpp)
target_link_libraries(test_decompress ${PROJECT_NAME} ${METIS_LIBRARY} GSL::gsl SZ3::SZ3)

add_executable(test_comp_decomp_cp test_comp_decomp_cp.cpp)
target_link_libraries(test_comp_decomp_cp ${PROJECT_NAME} ${METIS_LIBRARY} SZ3::SZ3)

add_executable(test_reorder test_reorder.cpp)
target_link_libraries(test_reorder ${PROJECT_NAME})