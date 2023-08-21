package com.benjamin.sg_counting_neural_network;

import java.util.List;

import java.util.ArrayList;

public class Trainer {
	Network network ;
	DataManager dataManager ;
	List<Image> trainingData = new ArrayList<>(), testingData = new ArrayList<>() ;
	
	double learningRate ;
	int numEpochs ;
	public double performance ;

	
	// MNIST constructor
    public Trainer (double learningRate, int numEpochs, int nbLayers, int hiddenLayerSize, String trainPath, String testPath) {
    	dataManager = new DataManager(trainPath, testPath) ;
    	this.network = new Network (nbLayers, 28*28, hiddenLayerSize, 10, "SIGMOID", "SQUARE_ERROR") ;
    	this.numEpochs = numEpochs ;
    	trainingData = dataManager.trainingData ;
    	testingData = dataManager.testData ;
    }
    
    // FF and gradient descent
    public void train () throws Exception {
    	network.InitializeNetwork() ;
    	double nablawhatever = learningRate/numEpochs ;
    	for (int i = 0 ; i < numEpochs ; i++) {
    		network.feedForward(trainingData.get(i).getImg());
    		// gradient descent
    		double[][][] costGradient_w = network.costGradient_w (trainingData.get(i).getLabelList()) ;
    		for (int l = 0 ; l < network.nbLayers ; l ++) {
    			for (int n = 0 ; n < network.Layers[l].layerSize ; n++) {
    				network.Layers[l].neurons[n].bias -= network.errorMat[l][n] * nablawhatever ;
    				for (int w = 0 ; w < network.Layers[l].neurons[n].weights.length ; w ++) {
    					network.Layers[l].neurons[n].weights[w] -= costGradient_w[l][n][w] * nablawhatever;
    				}
    			}
    		}
    	}
    }
    
    public void test () throws Exception {
    	for (int i = 0 ; i < testingData.size() ; i ++) {
    		network.feedForward(testingData.get(i).getImg());
    		performance += trainingData.get(i).getLabel()
    				== maxValueIndex(network.output) ? 1 : 0 ;
    	}
    	performance /= testingData.size() ;
    }
    
    private int maxValueIndex(double[] array) {
    	int max = 0 ;
    	for (int i = 0 ; i < array.length ; i ++) {
    		max = array[i] > max ? i : max ;
    	}
    	return max ;
    }
}
