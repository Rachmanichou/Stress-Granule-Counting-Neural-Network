package com.benjamin.sg_counting_neural_network;

import net.imglib2.img.Img;
import net.imglib2.type.numeric.real.FloatType;

public class Image {
	private Img<FloatType> img ;
	private int label ;
	
	public Image (Img<FloatType> img, int label) {
		this.img = img ;
		this.label = label ;
	}
	
	public Img<FloatType> getImg () {
		return img ;
	}
	
	public int getLabel () {
		return label ;
	}
	
	// if label = 3 then labelList = {0,0,0,1.0,0,0,0,0,0,0,0}
	public double[] getLabelList() {
		double[] labelList = new double[10] ;
		labelList[label] = 1.0 ;
		
		return labelList ;
	}
}
