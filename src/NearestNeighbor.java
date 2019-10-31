/**
 * Class for running the Nearest Neighbor algorithm
 * @author Abhijit Adhikary
 * 
 * Nearest Neighbor Algorithm Reference: https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm
 */
public class NearestNeighbor {
    
    // final variables
    private final double HIGH_DISTANCE = Double.POSITIVE_INFINITY;
    private final int NEGATIVE_VALUE = -99999;
    private final int TOTAL_FOLDS = 2;
    private final int TOTAL_LABEL_CLASSES = 10;
    private final int LABELS_PER_SAMPLE = 1;
    private final boolean PRINT_EACH_PREDICTION = false;
    private final boolean PRINT_CONFUSION_MATRIX = true;
    
    // train and test files
    private String trainFileName;
    private String testFileName;
    
    // arrays containing all the data from the dataset
    private int[][] trainData;
    private int[][] testData;
    
    // extracted feature array of the dataset
    private int[][] trainFeatures;
    private int[][] testFeatures;
    
    // extracted label array of the dataset
    private int[][] trainLabels;
    private int[][] testLabels;
    
    // total number of samples
    private int totalTrainSamples;    
    private int totalTestSamples;    
    
    // total number of attrubutes per sample
    private int totalTrainAttributes;
    private int totalTrainFeatures;
    private int totalTestAttributes;
    private int totalTestFeatures;
    
    // accuracy per fold
    private double[] foldAccuracy;
    // overall accuracy
    private double overallAccuracy;
    // controls whether to initialize the overallAccuracy variable
    private boolean foldAccuracyInilization;
    
    // matrix containing the predicted values for each sample
    int[][] predictedMatrix;
    
    // confusion matrix
    int[][] confusionMatrix;
    
    // error of each label class in the confusion matrix
    int[] confusionMatrixErrorArray;
    
    /**
     * Constructor for NearestNeighbor class
     * @param trainFileName Name of the training file
     * @param testFileName Name of the test file
     */
    public NearestNeighbor(String trainFileName, String testFileName) {
        this.trainFileName = trainFileName;
        this.testFileName = testFileName;
        foldAccuracyInilization = false;
    }
    
    /**
     * Starts the nearest neighbor classification. By default runs for two folds and calculates the euclidian distance
     */
    public void run() {
        System.out.println("\n\n----------------------------------------------------------------------------");
        System.out.println("\tRunning NEAREST NEIGHBOR CLASSIFIER ON 8x8 MNIST DATASET");
        System.out.println("----------------------------------------------------------------------------");
        
        runAllFolds();
        calculateOverallAccuracy();
        
        System.out.println("\n\nOver " + TOTAL_FOLDS + " Folds:");
        System.out.println("--------------------------");
        System.out.printf("Overall Accuracy: %.5f", overallAccuracy);
        System.out.println(" %\n");
        
    }
    
    /**
     * Method for running all the folds
     */
    private void runAllFolds() {
        for (int fold = 0; fold < TOTAL_FOLDS; fold++) {
            System.out.println("\nFold " + (fold + 1) + " Results:");
            System.out.println("--------------------------");
            
            init(trainFileName, testFileName);
            foldAccuracy[fold] = testAccuracy();
            
            if (PRINT_CONFUSION_MATRIX) {
                buildConfusionMatrix();
                printConfusionMatrix();
            }
            
            swapTrainAndTestFile();
        }
    }
    
    /**
     * Initializes all the required variables for the training and testing data
     * @param TRAIN_FILE_NAME
     * @param TEST_FILE_NAME 
     */
    private void init(String TRAIN_FILE_NAME, String TEST_FILE_NAME) {
        
        // create FileScanner instances for train and test file
        FileScanner trainScanner = new FileScanner(TRAIN_FILE_NAME);
        FileScanner testScanner = new FileScanner(TEST_FILE_NAME);
        
        // get all the data from the datasets
        trainData = trainScanner.getDataArray();
        testData = testScanner.getDataArray();
        
        // separately stroe the features and attributes
        totalTrainSamples = trainScanner.getTotalSamples();
        totalTrainAttributes = trainScanner.getTotalAttributes();
        totalTrainFeatures = totalTrainAttributes - LABELS_PER_SAMPLE;
        
        totalTestSamples = testScanner.getTotalSamples();
        totalTestAttributes = testScanner.getTotalAttributes();
        totalTestFeatures = totalTrainAttributes - LABELS_PER_SAMPLE;
        
        if (totalTrainFeatures != totalTestFeatures) {
            System.out.println("Training and test feature dimentions don't match up. Please try with a correct dataset");
            System.exit(0);
        }
        
        // initialize the variables
        trainFeatures = new int[totalTrainSamples][totalTrainFeatures];
        trainLabels = new int[totalTrainSamples][LABELS_PER_SAMPLE];
        testFeatures = new int[totalTestSamples][totalTestFeatures];
        testLabels = new int[totalTestSamples][LABELS_PER_SAMPLE];
        
        // initialize the foldAccuracy for the first run
        if (!foldAccuracyInilization) {
            foldAccuracy = new double[TOTAL_FOLDS];
            foldAccuracyInilization = true;
        }
        predictedMatrix = new int[totalTrainSamples][LABELS_PER_SAMPLE];
        confusionMatrix = new int[TOTAL_LABEL_CLASSES][TOTAL_LABEL_CLASSES];
        confusionMatrixErrorArray = new int[TOTAL_LABEL_CLASSES];
        
        extractFeaturesAndLabels();
    }
    
    /**
     * Extracts features and labels from the input data
     */
    private void extractFeaturesAndLabels() {
        for (int sample = 0; sample < totalTrainSamples; sample++) {
            for (int feature = 0; feature < totalTrainAttributes; feature++) {
                if (feature == totalTrainAttributes - LABELS_PER_SAMPLE) {
                    trainLabels[sample][0] = trainData[sample][feature];
                    testLabels[sample][0] = testData[sample][feature];
                } else {
                    trainFeatures[sample][feature] = trainData[sample][feature];
                    testFeatures[sample][feature] = testData[sample][feature];
                }
            }
        }
    }

    /**
     * Calculates overall accuracy of all the folds
     */
    private void calculateOverallAccuracy() {
        double totalAccuracy = 0;
        for (int fold = 0; fold < TOTAL_FOLDS; fold++) {
            totalAccuracy += foldAccuracy[fold];
        }
        overallAccuracy = totalAccuracy / TOTAL_FOLDS;
    }
    
    /**
     * Swaps the training and test data
     */
    private void swapTrainAndTestFile() {
        String temp = trainFileName;
        trainFileName = testFileName;
        testFileName = temp;
    }
    
    /**
     * Returns the Euclidian distance between two samples
     * @param trainSample Sample from the train set
     * @param testSample Sample from the test set
     * @return Euclidian distance between the samples
     */
    private double getEuclidianDistance(int trainSample, int testSample) {
        int sum = 0;
        
        for (int feature = 0; feature < totalTrainFeatures; feature++) {
            sum += Math.pow(trainFeatures[trainSample][feature] - testFeatures[testSample][feature], 2);
        }
        
        double distance = Math.sqrt(sum);
        return distance;
    }
    
    /**
     * Returns the nearest neighbor of the test sample
     * @param testSample Sample from the test set
     * @return The nearest neighbor of the test sample
     */
    private int getNearestNeighbor(int testSample) {
        double shortestDistance = HIGH_DISTANCE;
        int nearestSample = NEGATIVE_VALUE;
        
        for (int trainSample = 0; trainSample < totalTrainSamples; trainSample++) {
            double distance = getEuclidianDistance(trainSample, testSample);
            if (distance < shortestDistance) {
                shortestDistance = distance;
                nearestSample = trainSample;
            }
        }

        return trainLabels[nearestSample][0];
    }
    
    /**
     * Returns the accuracy on the test set for 1 fold
     * @return 
     */
    private double testAccuracy() {
        int correct = 0;
        int incorrect = 0;
        
        for (int sample = 0; sample < totalTrainSamples; sample++) {
            int actualLabel = testLabels[sample][0];
            
            int predictedLabel = getNearestNeighbor(sample);
            predictedMatrix[sample][0] = predictedLabel;
            // Prints each sample's prediction
            if (PRINT_EACH_PREDICTION) {
                System.out.print("Sample:\t" + sample + "\t||\tActual:\t" + actualLabel + "\t||\tPredicted:\t" + predictedLabel);
                if (actualLabel != predictedLabel) {
                    System.out.print(" <--------------------------------- (Misclassified)");
                }
                System.out.println();
            }
            
            if (actualLabel == predictedLabel) {
                correct++;
            } else {
                incorrect++;
            }
        }
        
        double accuracy = (double)correct / totalTrainSamples * 100;
        
        System.out.println("Total Samples:\t" + totalTrainSamples);
        System.out.println("Correct:\t" + correct);
        System.out.println("Incorrect:\t" + incorrect);
        System.out.printf("Accuracy:\t%.5f", accuracy);
        System.out.println(" %");
        
        return accuracy;
    }
    
    /**
     * Builds the confusionMatrix and confusionMatrixErrorArray variables
     */
    private void buildConfusionMatrix() {
        for (int row = 0; row < TOTAL_LABEL_CLASSES; row++) {
            for (int col = 0; col < TOTAL_LABEL_CLASSES; col++) {
                confusionMatrix[row][col] = 0;
            }
        }
        
        for (int sample = 0; sample < totalTrainSamples; sample++) {
            int actualLabel = testLabels[sample][0];
            int predictedLabel = predictedMatrix[sample][0];
            
            confusionMatrix[predictedLabel][actualLabel] += 1;
        }
        
        for (int col = 0; col < TOTAL_LABEL_CLASSES; col++) {
            for (int row = 0; row < TOTAL_LABEL_CLASSES; row++) {
                if (col != row) {
                    confusionMatrixErrorArray[col] += confusionMatrix[row][col];
                }
            }
        }
    }
    
    /**
     * Prints the confusion matrix and error per class of labels
     */
    private void printConfusionMatrix() {
        System.out.println("\n\nConfusion Matrix\n");
        for (int labelHeader = 0; labelHeader < TOTAL_LABEL_CLASSES; labelHeader++) {
            System.out.print("\t" + labelHeader);   
        }
        
        System.out.println("\n-----------------------------------------------------------------------------------");

        for (int row = 0; row < TOTAL_LABEL_CLASSES; row++) {
            System.out.print(row + " ->\t");

            for (int col = 0; col < TOTAL_LABEL_CLASSES; col++) {
                System.out.print(confusionMatrix[row][col] + "\t");
            }
            System.out.println("\n");
        }
        System.out.println("-----------------------------------------------------------------------------------");
        
        System.out.print("Error->");
        for (int errorOnLabel = 0; errorOnLabel < TOTAL_LABEL_CLASSES; errorOnLabel++) {
            System.out.print("\t" + confusionMatrixErrorArray[errorOnLabel]);
        }
        System.out.println("\n");
    }
}