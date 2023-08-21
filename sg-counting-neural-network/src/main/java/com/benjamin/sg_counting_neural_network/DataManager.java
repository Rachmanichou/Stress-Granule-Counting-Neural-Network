package com.benjamin.sg_counting_neural_network;

import java.util.List;
import java.util.ArrayList;
import java.util.Random;
import java.io.BufferedReader;
import java.io.FileReader;

import io.scif.img.IO;
import io.scif.img.ImgIOException;
import io.scif.img.SCIFIOImgPlus;
import net.imglib2.Cursor;
import net.imglib2.img.Img;
import net.imglib2.type.numeric.real.FloatType;

import net.imglib2.img.array.ArrayImgFactory;


// can be used for training data (use the batched constructor) or for data to be analyzed
public class DataManager {
	private List<SCIFIOImgPlus<FloatType>> scifioImgs ;
	public List<Img<FloatType>> imgs ;
	public List<List<Img<FloatType>>> batches ;
	
	private final int rows = 28, cols = 28 ;
	//private final int BIT_DEPTH = 8 ;
	public List<Image> trainingData = new ArrayList<Image>(), testData = new ArrayList<Image>();
    
    public DataManager (String dataFilePath, String labelFilePath, int batchSize) throws ImgIOException {
    	// add all image files to the list
    	System.out.println("Loading data...\n") ;
    	scifioImgs = IO.openAll(dataFilePath, new FloatType()) ;
    	
    	// convert to the less heavy Img<T> type and randomly organize into batches
    	int nbBatches = (scifioImgs.size() + batchSize - 1) / batchSize ; // ceil(a/b) = floor(a/b + 1 - 1/b)
    	int random = 0 ;
    	Random rand = new Random() ;
    	
    	batchSize = batchSize > scifioImgs.size() ? scifioImgs.size() : batchSize ;
    	for (int i = 0 ; i < nbBatches ; i++) {
			// handle when batch size doesn't divide the amount of data
    		batchSize = batchSize > scifioImgs.size() ? scifioImgs.size() : batchSize ;
    		for (int j = 0 ; j < batchSize ; j++) {
    			random = rand.nextInt(scifioImgs.size()) ;
    			batches.get(i).add(scifioImgs.get(random).getImg()) ;
    			scifioImgs.remove(random) ;
    		}
    	}
    	System.out.println("Finished loading data.") ;
    }
    
    // MNIST constructor
    public DataManager (String trainingPath, String testPath) {
    	System.out.println("Loading data...\n") ;
    	trainingData = dataReader(trainingPath) ;
    	testData = dataReader(testPath) ;
    	System.out.println("Data loaded.\n") ;
    }
    
    public List<Image> dataReader (String path) {
    	List<Image> dataList = new ArrayList<Image> () ;
    	// open training data stored as CSV
        try (BufferedReader dataReader = new BufferedReader(new FileReader(path))) {
            String line;

            while((line = dataReader.readLine()) != null){
                String[] lineItems = line.split(",");

                Img<FloatType> img = new ArrayImgFactory<> (new FloatType ()).create(rows,cols) ;
                Cursor<FloatType> cursor = img.cursor() ;
                
                int label = Integer.parseInt(lineItems[0]);
                int i = 1;
                // Create an <code>Img<\code> based on the csv's data and store it along with its label in an <code>Image<\code>
                /* typing is a little off so here's a recap: csv stores as "int", 
                 * data manager opens it as float, network accesses it as double */
                while (cursor.hasNext()) {
                	cursor.fwd() ;
                	cursor.get().set((float) Integer.parseInt(lineItems[i]));
                	i ++ ;
                }
                dataList.add(new Image(img, label));
                //System.out.println(trainingData.get(i-1).getLabel()); makes the program crash...
            }        
        } catch (Exception e){
            throw new IllegalArgumentException("Data not found at " + path);
        }
        return dataList ;
    }
}
    
