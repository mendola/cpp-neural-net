#include "net.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <fstream>
#include <time.h>
#include <string>
/*************** Function Definitions for Class Net ******************/
Net::Net(const std::vector<unsigned> &netStructure, double Eta, double Alpha){
  srand(time(NULL));
  unsigned numLayers = netStructure.size();
  m_recentAverageSmoothingFactor = 0.25;
  m_nOutputs = netStructure.back();

  for (unsigned layerNum = 0; layerNum < numLayers; layerNum++){
    // Create Layer
    m_layers.push_back(Layer());

    // 0 outputs from output neuron
    unsigned numOutputs;
    if(layerNum==netStructure.size()-1){
      numOutputs = 0;
    }else{
      numOutputs = netStructure[layerNum+1];
    }

    // Add Neurons to new layer (one extra neuron in each layer for offset)
    for(unsigned neuronNum = 0; neuronNum <= netStructure[layerNum]; neuronNum++){
      m_layers.back().push_back(Neuron(numOutputs,neuronNum,Eta, Alpha));
      //std::cout<<"Created Neuron. Layer: "<<layerNum<<"\tNeuron: "<<neuronNum<<std::endl;
    }
    //Set Bias to 1
    m_layers.back().back().setOutputVal(-1.0);
  }
}

void Net::feedForward(const std::vector<double> &inputVals){
  assert(inputVals.size() == m_layers[0].size() - 1);
  // Set input layer vals
  for (unsigned i = 0; i<inputVals.size(); i++){
    m_layers[0][i].setOutputVal(inputVals[i]);
  }

  // Feed Forward
  for(unsigned layerNum = 1; layerNum<m_layers.size();layerNum++){
    Layer &prevLayer = m_layers[layerNum-1];
    for(unsigned n = 0; n<m_layers[layerNum].size() - 1; n++){
      m_layers[layerNum][n].computeOutput(prevLayer);
    }
  }
}
void Net::backPropagate(const std::vector<double> &targetVals){
  Layer &outputLayer = m_layers.back();
  m_error = 0.0;

// std::cout<<"Output Layer: "<<std::endl;
  for(unsigned n = 0; n<outputLayer.size() - 1; n++){
    double delta = targetVals[n] - outputLayer[n].getOutputVal();
    m_error += delta*delta;
  }
  m_error /= (outputLayer.size()-1);
  m_error = sqrt(m_error);
//  std::cout<<"m_error: "<<m_error<<std::endl;
  // Recent average measurement
  m_recentAverageError =  (m_recentAverageError * m_recentAverageSmoothingFactor + m_error) \
                        / (m_recentAverageSmoothingFactor + 1.0);
  //std::cout<<"Recent Error: "<<m_recentAverageError<<std::endl;
  // Calculate output layer gradients
      //std::cout<<"output gradient: ";
  for (unsigned n = 0; n<outputLayer.size(); n++){
    outputLayer[n].calcOutputGradient(targetVals[n]);
  }

  // Calculate hidden layer gradients
  for(unsigned layerNum = m_layers.size() - 2; layerNum>0; layerNum--){
    Layer &hiddenLayer = m_layers[layerNum];
    Layer &nextLayer = m_layers[layerNum + 1];

    for(unsigned n = 0; n<hiddenLayer.size();n++){
      hiddenLayer[n].calcHiddenLayerGradients(nextLayer);
    }
  }

  // Update connection weights from output to input
  for(unsigned layerNum = m_layers.size() - 1; layerNum > 0; layerNum--){
    Layer &layer = m_layers[layerNum];
    Layer &prevLayer = m_layers[layerNum - 1];

    for(unsigned n = 0; n<layer.size() - 1; n++){
      layer[n].updateInputWeights(prevLayer);
    }
  }
}
void Net::getResults(std::vector<double> &resultVals){
  resultVals.clear();

  for(unsigned n = 0; n<m_layers.back().size() - 1; n++){
    resultVals.push_back(m_layers.back()[n].getOutputVal());
  }
}

void Net::SetTargets(unsigned short idx, std::vector<double> &vec){
  static unsigned prevIdx = 0;
  vec[prevIdx] = 0.0;
  vec[idx] = 1.0;
  prevIdx = idx;
}

double Net::classify(std::vector<double> output){
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

void Net::printWeightsNet(){
  for(unsigned l = 0; l < m_layers.size()-1; l++){
    std::cout<<"\nLayer "<<l<<": "<<std::endl;
    Layer currLayer = m_layers[l];
    for(unsigned n = 0; n < currLayer.size(); n++){
      std::cout<<"\n\tNeuron "<<n<<": "<<std::endl;
      Neuron currNeuron = currLayer[n];
      currNeuron.printWeightsNeuron();
    }
  }

}

void Net::TrainSGD(unsigned nEpochs, std::vector<std::vector<double> > &trainData, std::vector<unsigned short> &trainLabels){
  unsigned nSamples = trainData.size();

  std::vector<double> targets, results;
  results.resize(m_nOutputs);
  targets.resize(m_nOutputs);
  std::fill(targets.begin(),targets.end(), 0.0);

  for(unsigned epoch = 0; epoch < nEpochs; epoch++){
    unsigned sampleIdx[trainData.size()];

    // Create a random sampling of the dataset
    for(unsigned i = 0; i<nSamples; i++){
      sampleIdx[i] = rand() % nSamples;
    }

    for(unsigned img = 0; img < nSamples; img++){
      //std::cout<<"Training img number " <<img<<std::endl;
      SetTargets(trainLabels[sampleIdx[img]], targets);
      feedForward(trainData[sampleIdx[img]]);
      //PrintImg(trainData[img]);
      getResults(results);
      //PrintVals(results,"Results: ",10);
      //PrintVals(targets, "Targets: ",10);
      backPropagate(targets);
      //PrintRecentAvgError(myNet);
    }   
    //printWeightsNet(); 
  }
}

// Train until accuracy on test dataset stops increasing or maxEpochs is reached
double Net::TrainEarlyStopping(unsigned maxEpochs, unsigned patience, std::vector<std::vector<double> > &trainData, std::vector<unsigned short> &trainLabels, std::vector<std::vector<double> > &testData, std::vector<unsigned short> &testLabels){
  double acc;
  double maxAcc = 0;
  double epochsSinceIncrease = 0;
  std::vector<double> accList;
  for(unsigned epoch = 0; epoch < maxEpochs; epoch++){
    this->TrainSGD(1,trainData, trainLabels);
    acc = this->TestSGD(testData, testLabels);
    std::cout<<"\n\tEpoch " << epoch<< "\tAccuracy = "<<acc<<std::endl;
    accList.push_back(acc);
    if(maxAcc >= acc){
      epochsSinceIncrease++;
      if(epochsSinceIncrease > patience){
        return acc;
      }
    } else{
      maxAcc = acc;
      epochsSinceIncrease = 0;
    }
  }
  std::cout<<"MaxEpochs reached without converging."<<std::endl;
  return acc;
}

double Net::TestSGD(std::vector<std::vector<double> > &testData, std::vector<unsigned short> &testLabels){
  std::vector<double> results;
  results.resize(m_nOutputs);
  std::vector<unsigned short>::iterator itLabel = testLabels.begin();
  std::vector<std::vector<double> >::iterator itData = testData.begin();
  unsigned correct = 0;
  unsigned tot = 0;
  for(unsigned img = 0; img < testLabels.size(); img++){
    feedForward(*itData);
    getResults(results);
    double classifiedDigit = classify(results);
    if(classifiedDigit == *itLabel){
      correct++;
      //std::cout<<"CORRECT"<<std::endl;
    }else{
      //std::cout<<"INCORRECT"<<std::endl;
    }
    itLabel++;
    itData++;
    tot++;
  }

  double Accuracy = (double)correct / (double)tot;
  return Accuracy;
}

void Net::AdjustTrainingRate(double newEta, double newAlpha){
  for(unsigned l = 0; l < m_layers.size(); l++){
//    std::cout<<"\nLayer "<<l<<": "<<std::endl;
    Layer currLayer = m_layers[l];
    for(unsigned n = 0; n < currLayer.size(); n++){
//      std::cout<<"\n\tNeuron "<<n<<": "<<std::endl;
      Neuron currNeuron = currLayer[n];
      currNeuron.setEta(newEta);
      currNeuron.setAlpha(newAlpha);
    }
  }
}

unsigned Net::LoadWeights(const char* filepath){
  std::ifstream iFile(filepath);
  if(iFile.is_open()){
    for (unsigned layerNum = 0; layerNum < m_layers.size()-1; layerNum++){
      Layer currLayer = m_layers[layerNum];
      for(unsigned neuronNum = 0; neuronNum < currLayer.size(); neuronNum++){
        Neuron currNeuron = currLayer[neuronNum];
        for(unsigned w = 0; w <= currNeuron.m_outputConnections.size(); w++){
         // std::cout<<"Loading layer "<<layerNum<<" | Neuron " << neuronNum << " | Weight " << w;
          std::string newWeight;
          iFile >> newWeight;
          double newW = std::atof(newWeight.c_str());
	  std::cout<<newW<<std::endl;
          currNeuron.m_outputConnections[w].m_weight = newW;
        }
      }
    }
    return 0;
  }else{
    std::cout<<"Couldn't open output file."<<std::endl;
    return 1;
  }
}

unsigned Net::SaveWeights(const char* filepath){
  std::ofstream oFile(filepath);
  if(oFile.is_open()){
    for (unsigned layerNum = 0; layerNum < m_layers.size()-1; layerNum++){
      Layer currLayer = m_layers[layerNum];
      for(unsigned neuronNum = 0; neuronNum < currLayer.size(); neuronNum++){
        Neuron currNeuron = currLayer[neuronNum];
        for(unsigned w = 0; w <= currNeuron.m_outputConnections.size(); w++){
          std::cout<<"Saving layer "<<layerNum<<" | Neuron " << neuronNum << " | Weight " << w<<std::endl;
          oFile << currNeuron.m_outputConnections[w].m_weight<<"\n";
        }
      }
    }
    return 0;
  }else{
    std::cout<<"Couldn't open output file."<<std::endl;
    return 1;
  }
}
/***********Function Definitions for class Neuron ***********************/
Neuron::Neuron(const unsigned numOutputs, const unsigned index, double Eta, double Alpha){
  m_neuronIndex = index;
  eta = Eta;
  alpha = Alpha;
  for(unsigned connectionNum = 0; connectionNum < numOutputs; connectionNum++){
    m_outputConnections.push_back(Connection());
  }
}

void Neuron::computeOutput(Layer &prevLayer){
  double sum = 0.0;
  // Compute weighted sum of inputs
  for(unsigned n = 0; n<prevLayer.size(); n++){
    sum += (prevLayer[n].getOutputVal() * prevLayer[n].m_outputConnections[m_neuronIndex].m_weight);
  }

  // Run transfer function on weighted sum
  m_outputVal = transferFunction(sum);
}

double Neuron::transferFunction(const double in){
  //return tanh(in);
  return (double)(1.0 / (1.0 + std::exp(-1*in)));
}

double Neuron::deltaTransferFunction(const double in){
  //return (1.0 - in*in); tanh
  return (double)(transferFunction(in)*(1-transferFunction(in)));
}

// Eq BP1 (from Neural Networks and Deep Learning.com)
void Neuron::calcOutputGradient(double targetVal){
  double delta = targetVal - m_outputVal;
  m_gradient = delta * Neuron::deltaTransferFunction(m_outputVal);
  //std::cout<<"m_gradient: "<<m_gradient<<std::endl;
}

// Helper for BP2 (essentially computes error of this ne
double Neuron::sumDOW(const Layer &nextLayer){
  double sum = 0.0;

  // Sum weighted error from next layer (not bias node)
  for (unsigned n = 0; n < nextLayer.size() - 1; n++){
    sum += m_outputConnections[n].m_weight * nextLayer[n].m_gradient;
  }
  return sum;
}

// Eq BP2
void Neuron::calcHiddenLayerGradients(Layer &nextLayer){
  double dow = sumDOW(nextLayer);
  m_gradient = dow * Neuron::deltaTransferFunction(m_outputVal);
  //std::cout<<"Hidden Layer Gradient: "<<m_gradient<<std::endl;
}

void Neuron::updateInputWeights(Layer &prevLayer){
  for(unsigned n = 0; n<prevLayer.size(); n++){
    Neuron &neuron = prevLayer[n];
    double oldDeltaWeight = neuron.m_outputConnections[m_neuronIndex].m_deltaWeight;

    double newDeltaWeight = 
      //Individual input, scaled by gradint and train rate:
      // Eta scales the "learning rate" and alpha gives momentum from previous updates
      eta * neuron.getOutputVal() * m_gradient + alpha * oldDeltaWeight;

    neuron.m_outputConnections[m_neuronIndex].m_deltaWeight = newDeltaWeight;
    neuron.m_outputConnections[m_neuronIndex].m_weight += newDeltaWeight;
  }
}

void Neuron::printWeightsNeuron(){
  for(unsigned w = 0; w <= m_outputConnections.size(); w++){
    std::cout<<m_outputConnections[w].m_weight<<" ";
  }
}


