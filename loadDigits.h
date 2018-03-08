#ifndef LOAD_DIGITS_H
#define LOAD_DIGITS_H
#include <fstream>
#include <vector>
#include <string>

void ReadMNIST(int NumberOfImages, int DataOfAnImage,std::vector<std::vector<double> > &arr, const char* path);
void ReadLabelsMNIST(int NumberOfImages,std::vector<unsigned short> &arr, const char* path);
unsigned IndexFromDouble(double input);

#endif //LOAD_DIGITS_H