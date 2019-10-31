import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
/**
 * Class for reading in datasets and preparing the data for further use
 * @author Abhijit Adhikary
 */
class FileScanner {
    
    private String fileName;
    private int[][] dataArray;
    private int totalSamples;    
    private int totalAttrubutes;
    private ArrayList<int[]> inputList;
    public double[][] tempWeight;

    /**
     * Default constructor
     */
    public FileScanner() {
        
    }
    
    /**
     * Constructor for reading in specified file
     * @param fileName
     */
    public FileScanner(String fileName) {
        this.fileName = fileName;
        init();
    }
    
    /**
     * Reads in the data from the specified file and processes the data
     */
    private void init() {
        readInData(fileName);
        processInputData();
    }
    
    /**
     * Returns the array containing all the data
     * @return
     */
    public int[][] getDataArray() {
        return dataArray;
    }
    
    /**
     * Returns the total number of samples (rows)
     * @return
     */
    public int getTotalSamples() {
        return totalSamples;
    }
    
    /**
     * Returns the total number of attributes (columns)
     * @return
     */
    public int getTotalAttributes() {
        return totalAttrubutes;
    }
    
    /**
     * Method for reading in the data from a specified file
     * @param fileName
     */
    private void readInData(String fileName) {
      
        BufferedReader reader = null;
      
        try {
            reader = new BufferedReader(new FileReader(fileName));
            String inputLine;
            
            inputList = new ArrayList<>();

            // takes input from file until reaches the very end
            while ((inputLine = reader.readLine()) != null) {
                  // triggers if the line is not blank
                if (!inputLine.equals("")) {
                  // replaces any length of blank space with just one blank space and then trims any leading or following blank spaces
                  String tempString = inputLine.replaceAll("\\s{2,}", " ").trim();
                  // splits the String where blank space is present and stores in the parts String
                    String[] parts = tempString.split(",");

                    int[] tempArray = new int[parts.length];

                    // converts each element of the parts String to double and stores it in the tempArray
                    for(int element = 0; element < parts.length; element++) {
                        tempArray[element] = Integer.parseInt(parts[element]);
                    }

                    // adds the tempArray to the inputList ArrayList
                    inputList.add(tempArray);
                }
            }
            
            // totalSamples variable with the number of samples
            totalSamples = inputList.size();
            // number of features (65)
            totalAttrubutes = inputList.get(0).length;
          
            // if the file has no data in it the program exits
            if (inputList.isEmpty()) {
                  System.out.println("\"" + fileName + "\" is Empty\nPlease enter a valid file");
                  System.exit(0);
            }

        } catch (IOException e) {
          // triggers if the specified file is not found
            System.out.println(e);
            System.out.println("Could not find a file with the name \"" + fileName + "\"");
            System.out.println("Please make sure the file name is entered correctly");
            System.exit(0);

        } finally {
            try {
                if (reader != null) {
                  // closes the data file
                  reader.close();
                }
            } catch (IOException e) {
                  // triggers if there is a problem closing the file
                System.out.println(e);
                System.exit(0);
            }
        }
    }
    
    /**
     * Method for processing the input data
     */
    private void processInputData() {
        dataArray = new int[totalSamples][totalAttrubutes];

        // converts the inputList to an array
        for (int sample = 0; sample < totalSamples; sample++) {
            dataArray[sample] = inputList.get(sample);
        }
    }
    
    /**
     * Prints all the data in the data array
     */
    public void printData() {
        for (int sample = 0; sample < totalSamples; sample++) {
            for (int feature = 0; feature < totalAttrubutes; feature++) {
                System.out.print(dataArray[sample][feature] + " ");
            }
            System.out.println();
        }
    }
    
    
    
    
    /**
     * Method for saving trained weights
     * @param label
     * @param trainedWeightVector
     * @throws IOException
     */
    public void saveWeights(int label, double[][] trainedWeightVector) throws IOException {   
        
        String weights = Double.toString(trainedWeightVector[0][0]);

        for (int weight = 1; weight < trainedWeightVector.length; weight++) {
            weights = weights + ", " + Double.toString(trainedWeightVector[weight][0]);
        }
        String fileName = Integer.toString(label) + ".csv";
        
        try (FileWriter fileWriter = new FileWriter("pretrainedWeights/" + fileName)) {
            fileWriter.write(weights);
            System.out.println("Weights for label " + label + " saved successfully as \"" + fileName+"\"");
        }
    }
    
    /**
     * Method for loading pre-trained weights of a specified label
     * @param label
     * @return
     */
    public double[][] loadWeights(int label) {
        String fileName = "pretrainedWeights/" + Integer.toString(label)+".csv";
        
        BufferedReader reader = null;
      
        try {
            reader = new BufferedReader(new FileReader(fileName));
            String inputLine;
            
//            if (reader.readLine() == null) {
//                System.out.println("Weight file empty");
//            }
            
            // takes input from file until reaches the very end
            while ((inputLine = reader.readLine()) != null) {
                  // triggers if the line is not blank
                if (!inputLine.equals("")) {
                  // replaces any length of blank space with just one blank space and then trims any leading or following blank spaces
                  String tempString = inputLine.replaceAll("\\s{2,}", " ").trim();
                  // splits the String where blank space is present and stores in the parts String
                    String[] parts = tempString.split(",");

                    tempWeight = new double[parts.length][1];
                    // converts each element of the parts String to double and stores it in the tempArray
                    for(int element = 0; element < parts.length; element++) {
                        tempWeight[element][0] = Double.parseDouble(parts[element]);
                    }
                }
            }
            
            if (tempWeight == null) {
            	System.out.println("File empty");
            }
        } catch (IOException e) {
          // triggers if the specified file is not found
            System.out.println(e);
            System.out.println("Could not find a file with the name \"" + fileName + "\"");
            System.out.println("Please make sure the file name is entered correctly");
//            System.exit(0);

        } finally {
            try {
                if (reader != null) {
                  // closes the data file
                  reader.close();
                }
            } catch (IOException e) {
                  // triggers if there is a problem closing the file
                System.out.println(e);
                System.exit(0);
            }
        }
        
        return tempWeight;
    }
}