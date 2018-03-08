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

  void SetTargets(unsigned short idx, std::vector<double> &vec){
    static unsigned prevIdx = 0;
    vec[prevIdx] = 0.0;
    vec[idx] = 1.0;
    prevIdx = idx;
  }
  void PrintImg(std::vector<double> in){
    unsigned count = 0;
    for(std::vector<double>::iterator it = in.begin(); it != in.end(); ++it){
        if(*it > 10){
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
  }
int main(){
  /* Load training Data */
  std::vector<std::vector<double> > trainData;
  ReadMNIST(trainSetSize,784,trainData,trainDataPath);

  /* Load Training Labels */
  std::vector<unsigned short> trainLabels;
  ReadLabelsMNIST(trainSetSize, trainLabels, trainLabelPath);

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
  unsigned numInputs = 28*28;
  unsigned numOutputs = 10;
  unsigned hiddenLayerSize = 30;
  unsigned numReps = 10000;
  std::vector<unsigned> netStructure;
  netStructure.push_back(numInputs);
  netStructure.push_back(hiddenLayerSize);
  netStructure.push_back(numOutputs);

  Net myNet(netStructure);

  std::vector<double> targets, results;
  results.resize(numOutputs);
  targets.resize(numOutputs);
  std::fill(targets.begin(),targets.end(), 0.0);

  std::vector<unsigned short>::iterator itLabel = trainLabels.begin();
  
  for(unsigned rep = 0; rep < numReps; rep++){
    for(unsigned img = 0; img < dataSize; img++){
      SetTargets(*itLabel, targets);
      myNet.feedForward(trainData[img]);
      //PrintImg(trainData[img]);
      myNet.getResults(results);
      PrintVals(results,"Results: ",10);
      PrintVals(targets, "Targets: ",10);
      myNet.backPropagate(targets);
      PrintRecentAvgError(myNet);
      itLabel++;
    }
  }
  return 0;
}