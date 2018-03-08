#include "xor.h"
#include <cstdlib>

Pair getPair(){
  Pair p;
  p.x1 = (double)rand()/(double)RAND_MAX;
  p.x2 = (double)rand()/(double)RAND_MAX;
  if(p.x1>0.5){
    p.x1 = 1.0;
  }else{
    p.x1 = 0.0;
  }
  if(p.x2 > 0.5){
    p.x2 = 1.0;
  }else{
    p.x2 = 0.0;
  }
  return p;
}

double getXor(Pair p){
  double out;
  if(p.x1 + p.x2 == 1.0){
    out = 1.0;
  } else{
    out = 0.0;
  }
  return out;
}

std::vector<Pair> getInputs(unsigned numInputs){
  std::vector<Pair> pairVec;
  pairVec.resize(numInputs);
  for(unsigned i = 0; i<numInputs; i++){
    pairVec[i] = getPair();
  }
  return pairVec;
}

std::vector<double> getOutputs(std::vector<Pair> &inputs){
  std::vector<double> outVec;
  outVec.resize(inputs.size());
  for(unsigned i=0; i<inputs.size(); i++){
    outVec[i] = getXor(inputs[i]);
  }
  return outVec;
}