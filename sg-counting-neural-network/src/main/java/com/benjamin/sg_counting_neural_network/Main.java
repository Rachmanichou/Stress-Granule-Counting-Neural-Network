package com.benjamin.sg_counting_neural_network;

public class Main {
	public static void main (String[] args) throws Exception {
		final String trainPath = "/home/criuser/Desktop/Stage_L3/Image Analysis/MNIST/mnist_train.csv", 
				testPath = "/home/criuser/Desktop/Stage_L3/Image Analysis/MNIST/mnist_test.csv" ;
		
		// Trainer (learningRate, number of Epochs, nb Layers, hidden layer Size, training data, test data)
		Trainer trainer = new Trainer(0.5, 10, 4, 3, trainPath, testPath) ;
		System.out.println("Training...\n") ;
		trainer.train() ;
		System.out.println("Training done. Testing...\n");
		trainer.test() ;
		System.out.println("Performances = " + trainer.performance) ;
	}
}
