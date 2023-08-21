package com.benjamin.sg_counting_neural_network;

import java.lang.Math ;

public class MathFunctions {
    public double sigmoid (double x) {
        return 1/(1+Math.pow(Math.E,-x)) ;
    }
    public double sigmoid_x (double x) {
        return sigmoid(x) * (1-sigmoid(x)) ;
    }
    public double ReLU (double x) {
        return Math.max(0, x) ;
    }
    public double ReLU_x (double x) {
        return (x>0)?1.0:0.0 ;
    }
    public double squareError (double a, double b) {
        return 0.5 * (a-b)*(a-b) ;
    }
    public double squareError_a (double a, double b) {
        return a-b ;
    }
}


