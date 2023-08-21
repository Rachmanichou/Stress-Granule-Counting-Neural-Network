package com.benjamin.sg_counting_neural_network;

import java.util.Random ;

public class Neuron {
    public double activation, bias ;
    public int nbOutputs ;
    public double[] weights ;
    Random rand = new Random() ;

    //Constructor for untrained network
    public Neuron (int nbOutputs) {
        this.activation = rand.nextDouble() ;
        this.nbOutputs = nbOutputs ;
        this.bias = rand.nextDouble() ;
        this.weights = new double[nbOutputs] ;

        for (int i = 0 ; i < nbOutputs ; i++) {
            this.weights[i] = rand.nextDouble() ;
        }
    }

    //Constructor for trained network
    public Neuron (double activation, int nbOutputs, double bias, double[] weights) {
        this.activation = activation ;
        this.nbOutputs = nbOutputs ;
        this.bias = bias ;
        this.weights = weights ;
    }
}
