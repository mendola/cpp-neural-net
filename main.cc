#include "net.h"
#include "iostream"
#include "loadDigits.h"

const char* testDataPath = "data/t10k-images-idx3-ubyte";
const char* testLabelPath = "data/t10k-labels-idx1-ubyte";
const char* trainDataPath = "data/train-images-idx3-ubyte";
const char* trainLabelPath = "data/train-labels-idx1-ubyte";
int testSetSize = 10000;
int trainSetSize = 60000;

int main(){
  std::vector<std::vector<double> > trainData;
  ReadMNIST(trainSetSize,784,trainData,trainDataPath);

  std::vector<unsigned short> trainLabels;
  ReadLabelsMNIST(trainSetSize, trainLabels, trainLabelPath);

  std::vector<unsigned short>::iterator itLabel = trainLabels.begin();

for(int img = 0; img < 10;  img++){ //trainData.size();
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

}

  std::vector<unsigned> netStructure;
  netStructure.push_back(3);
  netStructure.push_back(2);
  netStructure.push_back(1);

  std::vector<double> inputs;
  inputs.push_back(3.92);
  inputs.push_back(0.04);
  inputs.push_back(2.4);

  Net myNet(netStructure);

  myNet.feedForward(inputs);
  myNet.backPropagate(inputs);
  myNet.getResults(inputs);

  return 0;
}