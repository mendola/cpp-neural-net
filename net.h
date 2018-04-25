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
    static double randWeight() {return (double)rand()/(double)RAND_MAX/(double)5;} 
};

class Net{
  public:
    Net(const std::vector<unsigned> &netStructure, double Eta, double Alpha);
    void feedForward(const std::vector<double> &inputVals);
    void backPropagate(const std::vector<double> &targetVals);
    void getResults(std::vector<double> &resultVals);
    double getRecentAvgError() {return m_recentAverageError;}
    void TrainSGD(unsigned nEpochs, std::vector<std::vector<double> > &trainData, std::vector<unsigned short> &trainLabels);
    double TrainEarlyStopping(unsigned maxEpochs, unsigned patience, std::vector<std::vector<double> > &trainData, std::vector<unsigned short> &trainLabels, std::vector<std::vector<double> > &testData, std::vector<unsigned short> &testLabels);
    double TestSGD(std::vector<std::vector<double> > &testData, std::vector<unsigned short> &testLabels);
    void printWeightsNet();
    void AdjustTrainingRate(double newEta, double newAlpha);
    unsigned LoadWeights(const char* filepath);
    unsigned SaveWeights(const char* filepath);
  private:
    void SetTargets(unsigned short idx, std::vector<double> &vec);
    double classify(std::vector<double> output);
    std::vector<Layer> m_layers;  // m_layers[Layer #][Neuron #]
    double m_error;
    double m_recentAverageError;
    double m_recentAverageSmoothingFactor;
    unsigned m_nOutputs;
};


class Neuron{
  public:
    Neuron(const unsigned numOutputs, const unsigned index, double Eta, double Alpha);
    void setOutputVal(const double oVal) {m_outputVal = oVal;}
    double getOutputVal() {return m_outputVal;}
    void computeOutput(Layer &prevLayer);
    void calcOutputGradient(double targetVal);
    void calcHiddenLayerGradients(Layer &nextLayer);
    double sumDOW(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer); 
    void setAlpha(double a) {alpha = a;}
    void setEta(double e) {eta = e;}
    void printWeightsNeuron();
    double eta;  // Overal Net learning rate [0.0 -- 1.0]
    double alpha; // Momentum [0.0 -- 1.0]
    std::vector<Connection> m_outputConnections;
  private:
    unsigned m_neuronIndex;
    double m_outputVal;
    static double transferFunction(const double in);
    static double deltaTransferFunction(const double in);
    double m_gradient;
};


#endif //NET_H