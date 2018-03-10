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

void SetTargets(unsigned short idx, std::vector<double> &vec){
  static unsigned prevIdx = 0;
  vec[prevIdx] = 0.0;
  vec[idx] = 1.0;
  prevIdx = idx;
}

void PrintVals(std::vector<double> &in, const char* str, unsigned checkSize){
  std::cout<<str;
  for(unsigned i = 0; i<in.size(); i++){
    std::cout<<in[i]<<" ";
  }
  std::cout<<std::endl;
  assert(in.size() == checkSize);
}

double classify(std::vector<double> output){
  unsigned imax = 0;
  double valmax = 0;
  for(unsigned i = 0; i< output.size(); i++){
    if(output[i] > valmax){
      valmax = output[i];
      imax = i;
    }
  }
  return (double)imax;
}
  
int main(){
  /* Load training Data */
  std::vector<std::vector<double> > trainData;
  ReadMNIST(trainSetSize,784,trainData,trainDataPath);

  /* Load Training Labels */
  std::vector<unsigned short> trainLabels;
  ReadLabelsMNIST(trainSetSize, trainLabels, trainLabelPath);

  unsigned dataSize = 2000;
  unsigned numInputs = 28*28;
  unsigned numOutputs = 10;
  unsigned hiddenLayerSize = 30;
  unsigned numReps = 30;
  std::vector<unsigned> netStructure;
  netStructure.push_back(numInputs);
  netStructure.push_back(hiddenLayerSize);
  netStructure.push_back(numOutputs);

  double etaVals[5] = {0.1, 0.15, 0.25, 1.0, 3.0};
  double alphaVals[5] = {0.1, 0.2, 0.25, 3, 0.35};
  std::vector<std::vector<double> > accuracies;
  accuracies.resize(5);
  for(unsigned i = 0; i<5; i++){
    accuracies[i].resize(5);
  }
  
  //
  for(unsigned i = 0; i < 5; i++){
    for(unsigned j = 0; j< 5; j++){
      std::cout<<"Working on i="<<i<<"  j="<<j<<std::endl;
      Net myNet(netStructure,etaVals[i],alphaVals[j]);

      std::vector<double> targets, results;
      results.resize(numOutputs);
      targets.resize(numOutputs);
      std::fill(targets.begin(),targets.end(), 0.0);


      
      for(unsigned rep = 0; rep < numReps; rep++){
        std::vector<unsigned short>::iterator itLabel = trainLabels.begin();
        for(unsigned img = 0; img < dataSize; img++){
          SetTargets(*itLabel, targets);
          myNet.feedForward(trainData[img]);
          //PrintImg(trainData[img]);
          myNet.getResults(results);
          //PrintVals(results,"Results: ",10);
          //PrintVals(targets, "Targets: ",10);
          myNet.backPropagate(targets);
          //PrintRecentAvgError(myNet);
          itLabel++;
        }
      }


      /**************Test on test data********************/
        /* Load training Data */
      std::vector<std::vector<double> > testData;
      ReadMNIST(testSetSize,784,testData,testDataPath);

      /* Load Training Labels */
      std::vector<unsigned short> testLabels;
      ReadLabelsMNIST(testSetSize, testLabels, testLabelPath);
      unsigned correct = 0;
      unsigned tot = 0;
      std::vector<unsigned short>::iterator itLabel = testLabels.begin();
      for(unsigned img = 0; img < dataSize; img++){
        myNet.feedForward(testData[img]);
        myNet.getResults(results);
        double classifiedDigit = classify(results);
        if(classifiedDigit == testLabels[img]){
          correct++;
          std::cout<<"CORRECT"<<std::endl;
        }else{
          std::cout<<"INCORRECT"<<std::endl;
        }
        itLabel++;
        tot++;
      }

      double Accuracy = (double)correct / (double)tot;
      std::cout<<"\n\nAccuracy = "<<Accuracy<<std::endl;
      accuracies[i][j] = Accuracy;
        }
      }
  std::cout<<"Parameter Accuracy relationship: "<<std::endl;
  int imax = 0;
  int jmax = 0;
  double bestAccuracy = 0;
  for(int i = 0; i< 5; i++){
    for(int j = 0; j<5; j++){
      std::cout<<"Eta = "<<etaVals[i]<<". Alpha = "<<alphaVals[j]<<".  Accuracy = "<<accuracies[i][j]<<std::endl;
      if(accuracies[i][j]>bestAccuracy){
        imax = i;
        jmax = j;
        bestAccuracy = accuracies[i][j];
      }
    }
  }

  std::cout<<"Best parameters: Eta = "<<etaVals[imax] << "  Alpha = "<<alphaVals[jmax]<<std::endl;
  return 0;
}