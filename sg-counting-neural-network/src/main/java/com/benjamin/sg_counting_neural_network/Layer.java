package com.benjamin.sg_counting_neural_network;

import net.imglib2.img.Img;
import net.imglib2.Cursor;
import net.imglib2.type.numeric.real.FloatType;

public class Layer {
    public int layerSize, nextLayerSize ;
    public Neuron[] neurons ;
    public double[][] layerWeights ;

    // Constructor for untrained network
    public Layer (int layerSize, int nextLayerSize) {
        this.layerSize = layerSize ;
        this.nextLayerSize = nextLayerSize ;
        this.neurons = new Neuron[layerSize] ;
        layerWeights = new double[layerSize][nextLayerSize] ;
        
        for (int i = 0 ; i < neurons.length ; i++) {
            neurons[i] = new Neuron(nextLayerSize) ;
        }
    }
    
    // Constructor for trained network
    public Layer (int layerSize, int nextLayerSize,
    		double activation, double bias, double[] weights) {
        this.layerSize = layerSize ;
        this.nextLayerSize = nextLayerSize ;
        this.neurons = new Neuron[layerSize] ;
        layerWeights = new double[layerSize][nextLayerSize] ;
        
        for (int i = 0 ; i < neurons.length ; i++) {
            neurons[i] = new Neuron(activation, nextLayerSize, bias, weights) ;
        }
    }
    
    public double[][] getWeights () {
    	double[][] layerWeights = new double[layerSize][nextLayerSize] ;
    	for (int n = 0 ; n < layerSize ; n ++) {
    		for (int w = 0 ; w < nextLayerSize ; w ++) {
    			layerWeights[n][w] = neurons[n].weights[w] ;
    		}
    	}
    	return layerWeights ;
    }
    
    //When a new activation matrix is computed, it is looped through and each column is passed to associated layer.
    public void UpdateFF (double[] activMatCol) {
        for (int i = 0 ; i < neurons.length ; i++) {
            neurons[i].activation = activMatCol[i] ;
        }
    }
    public void UpdateFF (Img<FloatType> input) throws Exception {
    	try {
    		Cursor<FloatType> cursor = input.cursor() ;
    		int i = 0 ;
    		while (cursor.hasNext()) {
    			cursor.fwd() ;
    			neurons[i].activation = cursor.get().getRealDouble() ;
    			i ++ ;
    		}
    	} catch (Exception e) {
    		System.out.println(e + ": invalid input or first layer sizes!") ;
    	}
    }
}
