package com.benjamin.sg_counting_neural_network;

import java.util.Arrays;

import net.imglib2.img.Img;
import net.imglib2.type.numeric.real.FloatType;

public class Network {
    int nbLayers ; //Takes into account input and output layers
    int inputSize, outputSize, hiddenLayerSize, largestLayerSize ;
    Layer[] Layers ;
    String actFunction, costFunction ; // ReLU, SIGMOID

    Matrix matrix = new Matrix() ;
    MathFunctions functions = new MathFunctions() ;
    
    public double[][] errorMat ; // this matrix is used twice per training iteration. Make it accessible to avoid computing it twice
    public double[] output ;
    
    public Network (int nbLayers, int inputSize, int hiddenLayerSize, int outputSize, 
    		String actFunction, String costFunction) {
        this.nbLayers = nbLayers ;
        this.inputSize = inputSize ;
        this.hiddenLayerSize = hiddenLayerSize ;
        this.outputSize = outputSize ;
        largestLayerSize = inputSize > hiddenLayerSize ? inputSize : hiddenLayerSize ;
        largestLayerSize = outputSize > inputSize ? outputSize : inputSize ;
        this.Layers = new Layer[nbLayers] ;
        this.actFunction = actFunction ;
        this.costFunction = costFunction ;
        
        errorMat = new double[nbLayers][largestLayerSize] ;
        output = new double[outputSize] ;
    }

    public void InitializeNetwork() {
        int laySize, nextLaySize ;
        for (int i = 0 ; i < nbLayers ; i++) {
        //Special cases of outermost layers
            if (i == 0) {
                laySize = inputSize ;
                nextLaySize = hiddenLayerSize ;
            } 
            else if (i == nbLayers - 1) {
                laySize = outputSize ;
                nextLaySize = 0 ;
            }
            else if (i == nbLayers - 2) {
                laySize = hiddenLayerSize ;
                nextLaySize = outputSize ;
            }
            else {
                laySize = hiddenLayerSize ;
                nextLaySize = hiddenLayerSize ;
            }

            Layers[i] = new Layer (laySize, nextLaySize) ;
        }
    }

    public void feedForward (Img<FloatType> input) throws Exception {
        Layers[0].UpdateFF(input);
        double[][] actMat = activationMatrix(false) ;
        int ctr = 0 ;
        for (int i = 0 ; i < nbLayers-1 ; i++) {
            for (int j = 0 ; j < Layers[i].nextLayerSize ; j++) {
                double temp = Layers[i+1].neurons[j].activation ;
                Layers[i+1].neurons[j].activation = actMat[i+1][j] ;
                if (temp == Layers[i+1].neurons[j].activation) {
                	ctr++ ;
                	System.out.println(ctr) ;
                }
            }
        }
        //System.out.println("Layers[1].neurons[2].activation =" + Layers[1].neurons[2].activation + " Layers[1].neurons[2].weights[0] ="+Layers[1].neurons[2].weights[0]) ;
        output = actMat[nbLayers-1] ;
    }


    //_________________________________________________________________Backpropagation___________________________________________________________

    //[which layer][which neuron][which neurite]
    public double[][][] costGradient_w (double[] expectedOutput) {
        double[][][] costGrad_w = new double[nbLayers][largestLayerSize][largestLayerSize] ;
        errorMat = errorMatrix(expectedOutput) ;
        double[][] activations = getActivations() ;

        for (int l = 1 ; l < nbLayers ; l++) {
        	for (int j = 0 ; j < Layers[l-1].layerSize ; j++) {
        		for (int k = 0 ; k < Layers[l].layerSize ; k++) {
        			costGrad_w[l][j][k] = activations[l-1][j] *  errorMat[l][k]; // TODO check indices order...
        		}
        	}
        }
        return costGrad_w ;
    }

    private double[][] errorMatrix (double[] expectedOutput) {
        double[][] weight_T_x_errorMat = new double[nbLayers][largestLayerSize] ; //useful quantity
        double[][] transpose = new double[nbLayers][largestLayerSize] ; // useful quantity
        double[][] activationGrad_z = activationMatrix(true) ;
        
        //Base case on output layer
        errorMat[nbLayers-2] = matrix.naiveHadamard(costGradient_a(expectedOutput), shrinkColumn(activationGrad_z, nbLayers - 1)) ; // TODO check: changed errorMat[nb-1] to nb-2 since neurites plug on the but
        
        // neurons of layer l have neurites projecting to layer l+1. Last layer has no neurites
        for (int l = nbLayers - 3 ; l >= 0 ; l--) {
            transpose = matrix.transpose(Layers[l+1].getWeights()) ;
            matrix.mapMatrix( matrix.ijkMatMult(transpose, shrinkColumn(errorMat, l+1)) , weight_T_x_errorMat[l]) ; // weight_T_x_errorMat[l] = transpose × errorMat[l+1]
            matrix.mapMatrix( matrix.naiveHadamard(shrinkColumn(weight_T_x_errorMat,l),shrinkColumn(activationGrad_z,l)) , errorMat[l]) ; // error[l] = weight_T_x_errorMat[l] ⋅ activationGrad_z[l]
        }
        return errorMat ;
    }

    // is computed only on the last layer
    private double[] costGradient_a (double[] expectedOutput) {
        double[] costGrad_a = new double[outputSize] ;
        for (int i = 0 ; i < expectedOutput.length ; i++) {
	        switch (costFunction) {
	            case "SQUARE_ERROR":
	                costGrad_a[i] = functions.squareError_a(Layers[nbLayers-1].neurons[i].activation, expectedOutput[i]) ;
	                //System.out.println(costGrad_a[i]) ;
	        }
        }
        //System.out.print("\n");
        return costGrad_a ;
    }

//________________________________________ Utility ____________________________________________________________________________
    
    /* Loop through <code>layer.nextLayerSize * layer.layerSize<\code> neurites of the layer's 
     * <code>layer.layerSize<\code> neurons 
     * Compute the activation of the next layer's neurons based on the values held by the neurites
    */
    private double[][] activationMatrix (boolean computeGradient) {
        double[][] actMat = new double[nbLayers][largestLayerSize] ;
        double z = 0;
        for (int i = 0 ; i < nbLayers-1 ; i++) {
            for (int j = 0 ; j < Layers[i].layerSize ; j++) {
                for (int k = 0 ; k < Layers[i].nextLayerSize ; k++) {
                    z = Layers[i].neurons[j].weights[k] * Layers[i].neurons[j].activation + Layers[i].neurons[j].bias ;
                    if (!computeGradient)
                        {
                            switch (actFunction) {
                                case "ReLU":
                                    actMat[i+1][j] = functions.ReLU(z) ;
                                    break ;
                                case "SIGMOID":
                                    actMat[i+1][j] = functions.sigmoid(z) ;
                            } 
                    }
                    switch (actFunction) {
                        case "ReLU":
                            actMat[i+1][j] = functions.ReLU_x(z) ;
                            break ;
                        case "SIGMOID":
                            actMat[i+1][j] = functions.sigmoid_x(z) ;
                    } 
                }
            }
        }
        return actMat ;
    }
    
    private double[][] getActivations () {
        double[][] act = new double[nbLayers][largestLayerSize] ;
        for (int i = 0 ; i < nbLayers ; i++) {
            for (int j = 0 ; j < Layers[i].layerSize ; j++) {
                act[i][j] = Layers[i].neurons[j].activation ;
            }
        }
        return act ;
    }
    
    // Some columns have more rows than needed. Remove empty slots
    private double[] shrinkColumn (double[][] A, int columnIndex) {
    	if (A == null) {
    		return null ;
    	}
    	int nbRows = Layers[columnIndex].layerSize ;
    	double[] shrinked = new double[nbRows] ;
    	
    	for (int i = 0 ; i < nbRows ; i ++) {
    		shrinked[i] = A[columnIndex][i] ;
    	}
    	return shrinked ;
    }
}
