#include "net.h"
#include "iostream"
#include "loadDigits.h"
#include "cassert"
#include <cmath>

const char* testDataPath = "data/t10k-images-idx3-ubyte";
const char* testLabelPath = "data/t10k-labels-idx1-ubyte";
const char* trainDataPath = "data/train-images-idx3-ubyte";
const char* trainLabelPath = "data/train-labels-idx1-ubyte";
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
  unsigned hiddenLayerSize = 30;
  unsigned numEpochs = 50;
  std::vector<unsigned> netStructure;
  netStructure.push_back(numInputs);
  netStructure.push_back(hiddenLayerSize);
  netStructure.push_back(hiddenLayerSize);
  netStructure.push_back(numOutputs);

  double eta = 0.15;
  double alpha = 0.1;

  Net myNet(netStructure,eta,alpha);

  for(unsigned i = 0; i<5; i++){
    double acc = myNet.TrainEarlyStopping(50,15,trainData,trainLabels,testData,testLabels);
    std::cout<<"Accuracy after ith round of optimization: "<<acc<<std::endl;
    eta = 0.5*eta;
    alpha = 0.5*alpha;
    myNet.AdjustTrainingRate(eta,alpha);
  }

  /*
  double etaVals[5] = {0.1,0.15};
  double alphaVals[1] = {0.1};
  std::vector<std::vector<double> > accuracies;
  accuracies.resize(5);
  for(unsigned i = 0; i<5; i++){
    accuracies[i].resize(5);
  }
  
  
  for(unsigned i = 0; i < 2; i++){
    for(unsigned j = 0; j< 1; j++){
      std::cout<<"Working on i="<<i<<"  j="<<j<<std::endl;
      Net myNet(netStructure,etaVals[i],alphaVals[j]);
//      myNet.printWeightsNet
      double acc = myNet.TrainEarlyStopping(100,15,trainData,trainLabels,testData,testLabels);
      accuracies[i][j] = acc;  
    }
  }

  std::cout<<"Parameter Accuracy relationship: "<<std::endl;
  int imax = 0;
  int jmax = 0;
  double bestAccuracy = 0;
  for(int i = 0; i< 2; i++){
    for(int j = 0; j<1; j++){
      std::cout<<"Eta = "<<etaVals[i]<<". Alpha = "<<alphaVals[j]<<".  Accuracy = "<<accuracies[i][j]<<std::endl;
      if(accuracies[i][j]>bestAccuracy){
        imax = i;
        jmax = j;
        bestAccuracy = accuracies[i][j];
      }
    }
  }

  std::cout<<"Best parameters: Eta = "<<etaVals[imax] << "  Alpha = "<<alphaVals[jmax]<<std::endl;
 */
  return 0;
}