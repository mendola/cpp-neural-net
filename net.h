#ifndef NET_H
#define NET_H

#include <vector>
#include <iostream>
#include <cstdlib>

class Neuron;
typedef std::vector<Neuron> Layer;

class Connection{
  public:
    Connection(){
      m_weight = randWeight();
    }
    double m_weight;
    double m_deltaWeight;
  private:
    static double randWeight() {return (double)rand()/(double)RAND_MAX;}
};

class Net{
  public:
    Net(const std::vector<unsigned> &netStructure);
    void feedForward(const std::vector<double> &inputVals);
    void backPropagate(const std::vector<double> &targetVals);
    void getResults(std::vector<double> &resultVals);
    double getRecentAvgError() {return m_recentAverageError;}
  private:
    std::vector<Layer> m_layers;  // m_layers[Layer #][Neuron #]
    double m_error;
    double m_recentAverageError;
    double m_recentAverageSmoothingFactor;
};


class Neuron{
  public:
    Neuron(const unsigned numOutputs, const unsigned index);
    void setOutputVal(const double oVal) {m_outputVal = oVal;}
    double getOutputVal() {return m_outputVal;}
    void computeOutput(Layer &prevLayer);
    void calcOutputGradient(double targetVal);
    void calcHiddenLayerGradients(Layer &nextLayer);
    double sumDOW(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);
    void setAlpha(double a);
    void setEta(double e);
  private:
    static double eta;  // Overal Net learning rate [0.0 -- 1.0]
    static double alpha; // Momentum [0.0 -- 1.0]
    unsigned m_neuronIndex;
    double m_outputVal;
    std::vector<Connection> m_outputConnections;
    static double transferFunction(const double in);
    static double deltaTransferFunction(const double in);
    double m_gradient;
};


#endif //NET_H