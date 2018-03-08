#include "net.h"
#include "iostream"
#include "loadDigits.h"
#include "xor.h"
#include "cassert"

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

  void PrintRecentAvgError(Net &net){
    std::cout<<"Recent Error: "<< net.getRecentAvgError()<<std::endl;
  }

int main(){
  /* Load training Data */
  //std::vector<std::vector<double> > trainData;
  //ReadMNIST(trainSetSize,784,trainData,trainDataPath);

  /* Load Training Labels */
  //std::vector<unsigned short> trainLabels;
  //ReadLabelsMNIST(trainSetSize, trainLabels, trainLabelPath);

  //std::vector<unsigned short>::iterator itLabel = trainLabels.begin();

/*for(int img = 0; img < 10;  img++){ //trainData.size();
  int count = 0;
  unsigned short label = *itLabel;
  std::cout<<"Label: "<<label<<std::endl;
  itLabel++;
  for(std::vector<double>::iterator it = trainData[img].begin(); it != trainData[img].end(); ++it){
    if(*it > 5){
      std::cout<<"X";
    }else{
      std::cout<<" ";
    }
   // std::cout << *it;
    count++;
    if(count==28){
      std::cout<<"\n";
      count = 0;
    }
  }
  std::cout<<"\n\n";

}*/
  unsigned dataSize = 2000;
  std::vector<unsigned> netStructure;
  netStructure.push_back(2);
  netStructure.push_back(4);
  netStructure.push_back(1);

  Net myNet(netStructure);

  std::vector<Pair> inData = getInputs(dataSize);
  std::vector<double> outData = getOutputs(inData);

  std::vector<double> inputs, targets, results;
  inputs.resize(2);
  targets.resize(1);
  results.resize(1);

  for(unsigned rep = 0; rep<dataSize; rep++){
    inputs[0] = inData[rep].x1;
    inputs[1] = inData[rep].x2;
    targets[0] = outData[rep];

    //PrintVals(inputs,"Inputs: ",2);
    myNet.feedForward(inputs);
    myNet.getResults(results);
   // PrintVals(results,"Results: ",1);
   // PrintVals(targets, "Targets: ",1);
    myNet.backPropagate(targets);
    PrintRecentAvgError(myNet);
  }
  return 0;
}