#include "net.h"
#include "iostream"
#include "loadDigits.h"
#include "cassert"
#include <cmath>

const char* testDataPath = "data/t10k-images-idx3-ubyte";
const char* testLabelPath = "data/t10k-labels-idx1-ubyte";
const char* trainDataPath = "data/train-images-idx3-ubyte";
const char* trainLabelPath = "data/train-labels-idx1-ubyte";
const char* savepath = "model.txt";
int testSetSize = 10000;
int trainSetSize = 60000;


void PrintVals(std::vector<double> &in, const char* str, unsigned checkSize){
  std::cout<<str;
  for(unsigned i = 0; i<in.size(); i++){
    std::cout<<in[i]<<" ";
  }
  std::cout<<std::endl;
  assert(in.size() == checkSize);
}


  
int main(){
  /* Load training Data */
  std::vector<std::vector<double> > trainData;
  ReadMNIST(trainSetSize,784,trainData,trainDataPath);

  /* Load Training Labels */
  std::vector<unsigned short> trainLabels;
  ReadLabelsMNIST(trainSetSize, trainLabels, trainLabelPath);

  /* Load training Data */
  std::vector<std::vector<double> > testData;
  ReadMNIST(testSetSize,784,testData,testDataPath);

  /* Load Training Labels */
  std::vector<unsigned short> testLabels;
  ReadLabelsMNIST(testSetSize, testLabels, testLabelPath);

  unsigned dataSize = 2000;
  unsigned numInputs = 28*28;
  unsigned numOutputs = 10;
  unsigned hiddenLayerSize = 50;
  unsigned numEpochs = 100;
  std::vector<unsigned> netStructure;
  netStructure.push_back(numInputs);
  netStructure.push_back(hiddenLayerSize);
  netStructure.push_back(hiddenLayerSize);
  netStructure.push_back(numOutputs);

  double etavals[5] ={0.075,0.1,0.15,0.25,0.35};
  double alpha = 0.1;
  std::vector<double> accList;
for (unsigned j = 0; j< 5; j++){
  double eta = etavals[j];
  Net myNet(netStructure,eta,alpha);
  double acc;
  for(unsigned i = 0; i<3; i++){
    acc = myNet.TrainEarlyStopping(150,30,trainData,trainLabels,testData,testLabels);
    std::cout<<"Accuracy after ith round of optimization: "<<acc<<std::endl;
    eta = 0.5*eta;
    alpha = 0.5*alpha;
    myNet.AdjustTrainingRate(eta,alpha);
  }
  accList.push_back(acc);
}

for(unsigned j = 0; j<5;j++){
  std::cout<<"Accuracy for j = "<<j<<": "<<accList[j]<<std::endl;
//  myNet.SaveWeights(savepath);
}
  return 0;
}
