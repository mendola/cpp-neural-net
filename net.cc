#include "net.h"
#include <cassert>
#include <cmath>

/*************** Function Definitions for Class Net ******************/
Net::Net(const std::vector<unsigned> &netStructure){
  unsigned numLayers = netStructure.size();
  m_recentAverageSmoothingFactor = 0.25;
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
      m_layers.back().push_back(Neuron(numOutputs,neuronNum));
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


/***********Function Definitions for class Neuron ***********************/
double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;
Neuron::Neuron(const unsigned numOutputs, const unsigned index){
  m_neuronIndex = index;
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

void Neuron::calcOutputGradient(double targetVal){
  double delta = targetVal - m_outputVal;
  m_gradient = delta * Neuron::deltaTransferFunction(m_outputVal);
  //std::cout<<"m_gradient: "<<m_gradient<<std::endl;
}

double Neuron::sumDOW(const Layer &nextLayer){
  double sum = 0.0;

  // Sum weighted error from next layer (not bias node)
  for (unsigned n = 0; n < nextLayer.size() - 1; n++){
    sum += m_outputConnections[n].m_weight * nextLayer[n].m_gradient;
  }
  return sum;
}

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
      eta * neuron.getOutputVal() * m_gradient + alpha * oldDeltaWeight;

    neuron.m_outputConnections[m_neuronIndex].m_deltaWeight = newDeltaWeight;
    neuron.m_outputConnections[m_neuronIndex].m_weight += newDeltaWeight;
  }
}

