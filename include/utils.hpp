#ifndef _UMC_UTILS_HPP
#define _UMC_UTILS_HPP

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>

namespace UMC{

// modified from TypeManager.c
// change return value and increment byteArray
void 
convertIntArray2ByteArray_fast_1b_to_result_sz(const unsigned char* intArray, size_t intArrayLength, unsigned char *& compressed_pos){
	size_t byteLength = 0;
	size_t i, j; 
	if(intArrayLength%8==0)
		byteLength = intArrayLength/8;
	else
		byteLength = intArrayLength/8+1;
		
	size_t n = 0;
	int tmp, type;
	for(i = 0;i<byteLength;i++){
		tmp = 0;
		for(j = 0;j<8&&n<intArrayLength;j++){
			type = intArray[n];
			if(type == 1)
				tmp = (tmp | (1 << (7-j)));
			n++;
		}
    	*(compressed_pos++) = (unsigned char)tmp;
	}
}

// modified from TypeManager.c
// change return value and increment compressed_pos
unsigned char * 
convertByteArray2IntArray_fast_1b_sz(size_t intArrayLength, const unsigned char*& compressed_pos, size_t byteArrayLength){
    if(intArrayLength > byteArrayLength*8){
    	printf("Error: intArrayLength > byteArrayLength*8\n");
    	printf("intArrayLength=%zu, byteArrayLength = %zu", intArrayLength, byteArrayLength);
    	exit(0);
    }
    unsigned char * intArray = NULL;
	if(intArrayLength>0) intArray = (unsigned char*)malloc(intArrayLength*sizeof(unsigned char));
	size_t n = 0, i;
	int tmp;
	for (i = 0; i < byteArrayLength-1; i++) {
		tmp = *(compressed_pos++);
		intArray[n++] = (tmp & 0x80) >> 7;
		intArray[n++] = (tmp & 0x40) >> 6;
		intArray[n++] = (tmp & 0x20) >> 5;
		intArray[n++] = (tmp & 0x10) >> 4;
		intArray[n++] = (tmp & 0x08) >> 3;
		intArray[n++] = (tmp & 0x04) >> 2;
		intArray[n++] = (tmp & 0x02) >> 1;
		intArray[n++] = (tmp & 0x01) >> 0;		
	}
	tmp = *(compressed_pos++);	
	for(int i=0; n < intArrayLength; n++, i++){
		intArray[n] = (tmp & (1 << (7 - i))) >> (7 - i);
	}		
	return intArray;
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
    std::cout << "Max value = " << max_val << ", min value = " << min_val << std::endl;
    std::cout << "Max error = " << max_err << ", pos = " << pos << std::endl;
    std::cout << "MSE = " << mse << ", PSNR = " << psnr << std::endl;
}

// katrina only
template <class T>
void print_statistics(const T * data_ori, const T * data_dec, size_t data_size, const T invalid_val){
    double max_val = INT8_MIN * 1.0;
    double min_val = INT8_MAX * 1.0;
    for(int i=0; i<data_size; i++){
        if(data_ori[i] > max_val && data_ori[i] != invalid_val) max_val = data_ori[i];
        if(data_ori[i] < min_val && data_ori[i] != invalid_val) min_val = data_ori[i];
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
    std::cout << "Max value = " << max_val << ", min value = " << min_val << std::endl;
    std::cout << "Max error = " << max_err << ", pos = " << pos << std::endl;
    std::cout << "MSE = " << mse << ", PSNR = " << psnr << std::endl;
}

template<typename Type>
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

template<typename Type>
void writefile(const char *file, Type *data, size_t num_elements) {
    std::ofstream fout(file, std::ios::binary);
    fout.write(reinterpret_cast<const char *>(&data[0]), num_elements * sizeof(Type));
    fout.close();
}

// normalize data to [0, 1]
inline double normalize_data(double data, double min, double range){
	return (data - min)/range;
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

}
#endif