#ifndef XOR_H
#define XOR_H

#include "vector"

struct Pair{
  double x1;
  double x2;
};

Pair getPair();
double getXor(Pair p);
std::vector<Pair> getInputs(unsigned numInputs);
std::vector<double> getOutputs(std::vector<Pair> &inputs);

#endif