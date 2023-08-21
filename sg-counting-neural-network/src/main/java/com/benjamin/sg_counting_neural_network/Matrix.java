package com.benjamin.sg_counting_neural_network;

import java.util.function.DoubleUnaryOperator;

public class Matrix {
    //Use with this syntax: map(matrix, d -> foo(d)) ;
    public void mapFunc (double[][] A, int start_i, int start_j, DoubleUnaryOperator fn) {
        for (int i = start_i ; i < A.length ; i ++) {
            for (int j = start_j ; j < A[0].length ; j++) {
                A[i][j] = fn.applyAsDouble(A[i][j]) ;
            }
        }
    }
    // fit a small vector or matrix into a larger one starting at (0,0).
    public void mapMatrix (double[] smaller, double[] larger) {
    	if (larger.length < smaller.length) {
    		return ;
    	}
    	for (int i = 0 ; i < smaller.length ; i ++) {
    		larger[i] = smaller[i] ;
    	}
    }

    public double[][] transpose (double[][] A) {
        int n = A.length, m = A[0].length ;
        if (n < 1 || m < 1) {
            System.out.println("Impossible Transposition. Invalid matrix! Has "+n+" columns and "+m+" rows.") ;
            return A ;
        }
        double[][] T = new double[m][n] ;
        for (int i = 0 ; i < n ; i ++) {
            for (int j = 0 ; j < m ; j++) {
                T[j][i] = A[i][j] ;
            }
        }
        return T ; 
    }
    // with matrices
    public double[][] naiveHadamard (double[][] A, double[][] B) {
        //A: n×m ; B: m×p ;
        int n = A.length, m = A[0].length ;
        if (n != B.length || m != B[0].length) {
            System.out.println("Impossible Hadamard. Non-valid matrix sizes!") ;
            return null ;
        }
        double[][] C = new double[n][m] ;
        for (int i = 0 ; i < n ; i ++) {
            for (int j = 0 ; j < m ; j ++) {
                C[i][j] = A[i][j]*B[i][j] ;
            }
        }
        return C ;
    }
    // with column vectors
    public double[] naiveHadamard (double[] A, double[] B) {
        //A: n×1 ; B: m×1 ;
        int n = A.length, m = B.length;
        if (n != m ) {
            System.out.println("Impossible Hadamard. Non-valid matrix sizes!") ;
            return null ;
        }
        double[] C = new double[n] ;
        for (int i = 0 ; i < n ; i ++) {
                C[i] = A[i]*B[i] ;
        }
        return C ;
    }
    // with two matrices
    public double[][] ijkMatMult (double[][] A, double[][] B) {
        //A: n×m ; B: m×p ;
        int n = A.length, m = B.length, p = B[0].length ;
        if (A[0].length != m) {
            System.out.println("Impossible product. Non-valid matrix sizes!") ;
            return null ;
        }
        double[][] C = new double[n][p] ;
        for (int i = 0 ; i < n ; i ++) {
            for (int j = 0 ; j < p ; j ++) {
                for (int k = 0 ; k < m ; k ++) {
                    C[i][j] += A[i][k]*B[k][j] ;
                }
            }
        }
        return C ;
    }
    // with a column vector and a matrix
    public double[] ijkMatMult (double[][] A, double[] B) {
        //A: n×m ; B: m×1 ;
    	int n = A.length, m = A[0].length ;
        if (m != B.length) {
            System.out.println("Impossible product. Non-valid matrix sizes!") ;
            return null ;
        }
        double[] C = new double[n] ;
        for (int i = 0 ; i < n ; i ++) {
            for (int j = 0 ; j < m ; j ++) {
                C[i] += A[i][j] * B[j] ; 
            }
        }
        return C ;
    }
}