import java.io.IOException;

/**
 * Class for classifying using Soft Margin Support Vector Machine
 * Algorithm reference: https://en.wikipedia.org/wiki/Support-vector_machine
 * @author Abhijit Adhikary
 * The algorithm by default trains on half of a dataset, validates on the other of the same dataset. 
 * The training is done on a separate dataset.
 * To run all ta two fold test run the runAllFolds() method
 * To print the classification result for each label set printTrainingResultsForLabel to true
 */
public class SVM{
    
    // variables for loading and saving weight vectors
    private final boolean SAVE_WEIGHTS = false;
    private final boolean LOAD_WEIGHTS = false;
    
    // hyperparameters
    private double learningRate = 0.005;
    private int totalEpochs = 100;
    private final int validationLabel = 1;
    
    // dataset specifications
    private final int LABELS_PER_SAMPLE = 1;
    private final int TOTAL_LABEL_CLASSES = 10;
    private final int TOTAL_BIAS = 1;
    private final int TOTAL_FOLDS = 2;
    
    private int currentFold;
    private int currentTrainDataset;
    
    private final double NEGATIVE_INFINITY = Double.NEGATIVE_INFINITY;
    private final Matrix matrix = new Matrix();

    private int startTrainIndex;
    private int endTrainIndex;
    
    private int startTestIndex;
    private int endTestIndex;
    
    // train and test files
    private final String DATASET_1_FILENAME;
    private final String DATASET_2_FILENAME;
    
    // arrays containing all the data from the dataset
    private int[][] trainData;
    private int[][] testData;
    
    // total number of attrubutes per sample
    private int totalTrainAttributes;
    private int totalTrainFeatures;
    private int totalTestAttributes;
    private int totalTestFeatures;
    
    // extracted feature array of the dataset
    private double[][] trainFeatures;
    private double[][] testFeatures;
    
    // extracted label array of the dataset
    private int[][] trainLabels;
    private int[][] testLabels;
    
    // total number of samples
    private int totalTrainSamples;    
    private int totalTestSamples;
    
    // array containing a temporary weights
    private double[][] tempWeight;
    // array containing the current weights
    private double[][][] weightArray;
    // variable used to avoid overwriting previously trained weights
    private  boolean weightInitialization = false;
    // stores the current training labels to +1 and others to -1
    private double[][] oneHotLabelsTemp;
    
    // stores the prediction confidence number for each label
    private double[][] predictionMatrix;
    // stores the label with the highest confidence number
    private int[][] predictedMatrix;
    // matrix containing which label is confused by which label
    private int[][] confusionMatrix;
    // errors per label of the confusion matrix
    private int[] confusionMatrixErrorArray;
    
    private double[] testResults = new double[TOTAL_FOLDS];
    private boolean testing = false;
    
    // print conditions
    private boolean printInitialParameters = false;
    private boolean printTrainingResultsForLabel = false;
    
    // learning rate and epoch array for exploring hyperparameters
    private double[] learningRateArray = {0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001};
    private int[] epochArray = {50, 100, 150, 200, 250, 300, 500, 700, 1000};
    
    // learning rate and epoch array for dataset 1
    private double[] leraningRateDataset1 = {0.01, 0.005, 1.00E-04, 0.1, 0.005, 5.00E-04, 0.005, 0.005, 0.001, 0.005};
    private int[] epochsDataset1 = {700, 100, 300, 700, 700, 300, 500, 100, 250, 100};
    
    // learning rate and epoch array for dataset 2
    private double[] leraningRateDataset2 = {0.005, 0.005, 0.01, 0.001, 0.1, 0.01, 0.005, 0.005, 0.005, 0.005};
    private int[] epochsDataset2 = {200, 250, 700, 500, 150, 700, 250, 150, 150, 100};

    /**
     * constructor for soft margin SVM
     * @param trainFileName
     * @param testFileName 
     */
    public SVM(String trainFileName, String testFileName) {
        this.DATASET_1_FILENAME = trainFileName;
        this.DATASET_2_FILENAME = testFileName;
    }
    
    /**
     * Trains, validates and tests for two folds
     */
    public void runAllFolds() {
        System.out.println("\n\n------------------------------------------------------------------------------------------");
        System.out.println("\tRunning Soft Margin Support Vector Machine Classifier on 8x8 MNIST dataset");
        System.out.println("------------------------------------------------------------------------------------------");
        
        for (int fold = 0; fold < TOTAL_FOLDS; fold++) {
            currentTrainDataset = fold + 1;
            currentFold = fold;
            
            System.out.println("\n\n\n------------------------------------------------------------------------------------------");
            System.out.println("\t\t\t\tCurrent Fold: " + (currentFold + 1));
            System.out.println("------------------------------------------------------------------------------------------\n\n\n");
            
            // train on dataset 2; test on dataset 1
            if (currentFold == 0) {
                trainAndValidateDataset1();
                testing = true;
                testOnDataset2();
                testing = false;
            }
            
            // train on dataset 2; test on dataset 1
            if (currentFold == 1) {
                trainAndValidateDataset2();
                testing = true;
                testOnDataset1();
                testing = false;
            }
        }
        
        double totalAccuracy = 0;
        for (int fold = 0; fold < TOTAL_FOLDS; fold++) {
            totalAccuracy += testResults[fold];
        }
        
        double finalAccuracy = totalAccuracy / TOTAL_FOLDS;
        System.out.println("*******************************************************************************************");
        System.out.println("*******************************************************************************************");
        System.out.printf("\n\n\t\t\t\tFold 1 Accuracy:\t%.2f", testResults[0]);
        System.out.println(" %");
        System.out.printf("\t\t\t\tFold 2 Accuracy:\t%.2f",testResults[1]);
        System.out.println(" %");
        System.out.print("\t\t\t\tAverage Accuracy:\t");
        System.out.printf("%.2f", finalAccuracy);
        System.out.println(" %\n\n");
        System.out.println("*******************************************************************************************");
        System.out.println("*******************************************************************************************\n\n\n"); 
    }
    
    /**
     * initializes all the necessary variables
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
        totalTrainFeatures = totalTrainAttributes - LABELS_PER_SAMPLE + TOTAL_BIAS;
        
        totalTestSamples = testScanner.getTotalSamples();
        totalTestAttributes = testScanner.getTotalAttributes();
        totalTestFeatures = totalTestAttributes - LABELS_PER_SAMPLE + TOTAL_BIAS;
        
        // initialize the variables
        trainFeatures = new double[totalTrainSamples][totalTrainFeatures];
        trainLabels = new int[totalTrainSamples][LABELS_PER_SAMPLE];
        testFeatures = new double[totalTestSamples][totalTestFeatures];
        testLabels = new int[totalTestSamples][LABELS_PER_SAMPLE];
        
        // extracts the features and labels and stores them in separate variables
        extractFeaturesAndLabels();
        
        // add an extra bias feature at the end of the feature array
        addBiasAtTheEnd();
        
        predictionMatrix = new double[totalTestSamples][TOTAL_LABEL_CLASSES];
        predictedMatrix = new int[totalTrainSamples][1];
        confusionMatrix = new int[TOTAL_LABEL_CLASSES][TOTAL_LABEL_CLASSES];
        confusionMatrixErrorArray = new int[TOTAL_LABEL_CLASSES];
        
        // initializes the weight array for the first time
        if (!weightInitialization) {
            weightArray = new double[TOTAL_LABEL_CLASSES][][];
            weightInitialization = true;
        }
        
        oneHotLabelsTemp = null;
        oneHotLabelsTemp = new double[1][totalTrainSamples];
        
        startTrainIndex = 0;
        endTrainIndex = totalTrainSamples;

        startTestIndex = 0;
        endTestIndex = totalTestSamples;
        
        if (printInitialParameters) {
            printInitialParameters();
        }
    }
    
    /**
     * Trainins on 2nd half of Dataset 1
     * Validates on 1st half of Dataset 1
     */
    private void trainAndValidateDataset1() {
        init(DATASET_1_FILENAME, DATASET_1_FILENAME);
        startTrainIndex = totalTrainSamples / 2;
        endTrainIndex = totalTrainSamples;
        totalTrainSamples = totalTrainSamples / 2;
        
        startTestIndex = 0;
        endTestIndex = totalTestSamples / 2;
        totalTestSamples = totalTestSamples / 2;
        
        printTrainingResultsForLabel = false;
        System.out.println("------------------------------------------------------------------------------------------");        
        System.out.println("\t\t\tTraining on 2nd half of Dataset 1");
        System.out.println("\t\t\tValidating on 1st half of Dataset 1");
        System.out.println("------------------------------------------------------------------------------------------");        
        trainAllSVM();
        testAllSVM();
        
//        testHyperParameters();
//        trainAndTestSVM(validationLabel);
    }
    
    /**
     * Trainins on 2nd half of Dataset 2
     * Validates on 1st half of Dataset 2
     */
    private void trainAndValidateDataset2() {
        init(DATASET_2_FILENAME, DATASET_2_FILENAME);
        startTrainIndex = totalTrainSamples / 2;
        endTrainIndex = totalTrainSamples;
        totalTrainSamples = totalTrainSamples / 2;
        
        startTestIndex = 0;
        endTestIndex = totalTestSamples / 2;
        totalTestSamples = totalTestSamples / 2;
        
        printTrainingResultsForLabel = false;
        System.out.println("------------------------------------------------------------------------------------------");        
        System.out.println("\t\t\tTraining on 2nd half of Dataset 2");
        System.out.println("\t\t\tValidating on 1st half of Dataset 2");
        System.out.println("------------------------------------------------------------------------------------------");

        trainAllSVM();
        testAllSVM();

//        testHyperParameters();
//        trainAndTestSVM(validationLabel);
    }
    
    /**
     * Tests on dataset 1
     */
    private void testOnDataset1() {
        init(DATASET_2_FILENAME, DATASET_1_FILENAME);

        startTestIndex = 0;
        endTestIndex = totalTestSamples;
        
        printTrainingResultsForLabel = false;
        System.out.println("------------------------------------------------------------------------------------------");
        System.out.println("\t\t\t\tTesting on Dataset 1");
        System.out.println("------------------------------------------------------------------------------------------");
        testAllSVM();
    }
    
    /**
     * Tests on dataset 2
     */
    private void testOnDataset2() {
        init(DATASET_1_FILENAME, DATASET_2_FILENAME);

        startTestIndex = 0;
        endTestIndex = totalTestSamples;
        
        printTrainingResultsForLabel = false;
        System.out.println("------------------------------------------------------------------------------------------");
        System.out.println("\t\t\t\tTesting on Dataset 2");
        System.out.println("------------------------------------------------------------------------------------------");
        testAllSVM();
    }
    
    /**
     * Explores a range of learning rates and epochs
     */
    public void testHyperParameters() {
        for (int epoch = 0; epoch < epochArray.length; epoch++) {
                totalEpochs = epochArray[epoch];
            for (int tempRate = 0; tempRate < learningRateArray.length; tempRate++) {
                learningRate = learningRateArray[tempRate];

                trainAndTestSVM(validationLabel);
            
                System.out.println();
            } 
        }

		/*
        for (int tempRate = 0; tempRate < lr_array.length; tempRate++) {
            LEARNING_RATE = lr_array[tempRate];
            for (int epoch = 50; epoch < 10001; epoch = epoch + 50) {
                totalEpochs = epoch;
                trainAndTestSVM(validationLabel);
            }
            System.out.println();
        } 
		*/
    }
    
    /**
     * trains and tests for a single label
     * @param label 
     */
    private void trainAndTestSVM(int label) {
        trainSVM(label);
        testSVM(label);
    }
    
    /**
     * trains for all labels -> tests for all labels
     */
    private void trainAndTestAllSVM() {
        for (int label = 0; label < TOTAL_LABEL_CLASSES; label++) {
            trainSVM(label);
        }
        
        for (int label = 0; label < TOTAL_LABEL_CLASSES; label++) {
            testSVM(label);
        }
    }
    
    /**
     * trains for all labels
     */
    private void trainAllSVM() {
        for (int label = 0; label < TOTAL_LABEL_CLASSES; label++) {
            trainSVM(label);
        }
    }
    
    /**
     * tests for all labels
     */
    private void testAllSVM() {
        
        for (int label = 0; label < TOTAL_LABEL_CLASSES; label++) {
            testSVM(label);
        }
        
        printConfidenceMatrix();
        buildDecisionMatrix();
//        printDecisionMatrix();

        buildConfusionMatrix();
        printConfusionMatrix();
//        printFinalClassification();
        printFinalAccuracy();
    }
    
    /**
     * trains SVM on a specific label
     * @param label 
     */
    private void trainSVM(int label) {
        tempWeight = null;
        tempWeight = matrix.generateZeros(totalTrainFeatures);
        oneHotEncodeTrainLabels(label);
        
        if (currentTrainDataset == 1) {
            learningRate = leraningRateDataset1[label];
            totalEpochs = epochsDataset1[label];
        }
        
        if (currentTrainDataset == 2) {
            learningRate = leraningRateDataset2[label];
            totalEpochs = epochsDataset2[label];
        }
        
        System.out.println("\tTraining for Label: " + label + "\tLearning Rate: " + learningRate + "\tTotal Epochs: " + totalEpochs);
//        System.out.println("---------------------\n");
        
        for (int epoch = 1; epoch <= totalEpochs; epoch++) {
            
            if (epoch % 100 == 0) {
//                System.out.println("Epoch: " + epoch);
            }
            
            for (int sample = startTrainIndex; sample < endTrainIndex; sample++) {
                
                // takes the current sample and convert it to a row vector
                double[][] currentSample = matrix.convertToRowMatrix(trainFeatures, sample);
                
                // multiply current sample feature with the weight vector
                double weightTimesFeatureRow = matrix.dot(currentSample, tempWeight)[0][0];
                
                // true label of the current sample
                double currentLabel = oneHotLabelsTemp[0][sample];
                // multiply the weightTimesFeatureRow with the current true label
                double predictionValue = weightTimesFeatureRow * currentLabel;
                
                // misclassified if predictionValue is less than 1
                boolean misclassified = predictionValue < 1;
                
                // regularizer variable to decrease the learning rate as the epochs increase
                double regularizer = 1D / epoch;

                // if misclassified then update weight
                if (misclassified) {
                    double[][] currentSampleFeaturesTimesLabel = matrix.elementwiseMultiply(currentSample, currentLabel);
                    double[][] currentSampleFeaturesTimesLabelTransposed = matrix.transpose(currentSampleFeaturesTimesLabel);
                    
                    double[][] regularizerTimesWeight = matrix.elementwiseMultiply(tempWeight, (-2D * regularizer));
                    
                    double[][] rightToEta = matrix.elementwiseAdd(currentSampleFeaturesTimesLabelTransposed, regularizerTimesWeight);
                    double[][] update = matrix.elementwiseMultiply(rightToEta, learningRate);
                    
                    // update weight
                    tempWeight = matrix.elementwiseAdd(tempWeight, update);

                } else {
                    double[][] regularizerTimesWeight = matrix.elementwiseMultiply(tempWeight, (-2D * regularizer));
                    double[][] update = matrix.elementwiseMultiply(regularizerTimesWeight, learningRate);
                    
                    // update weight
                    tempWeight = matrix.elementwiseAdd(tempWeight, update);
                }
            }
        }
//        System.out.println("Training Finished for label: " + label);
        // stores the trained weight in the weight array at the specified label index
        weightArray[label] = tempWeight;
        if (SAVE_WEIGHTS) {
            saveWeights(label);
        }
    }
    
    /**
     * tests SVM on a specific label
     * @param testLabel 
     */
    private void testSVM(int testLabel) {
        
        double[][] currentWeight = null;
        if (LOAD_WEIGHTS) {
            currentWeight = loadWeights(testLabel);
        } else {
            currentWeight = weightArray[testLabel];
        }
        
        if (currentWeight == null) {
            return;
        }
        
        int totalTestSamplesInLoop = 0;
        int totalTestLabels = 0;
        
        int totalError;
        int overallCorrect = 0;
        int currentLabelCorrect = 0;
        
        int underestimate = 0;
        int overestimate = 0;
        
        double accuracy;
        
//        System.out.println("\n-----------------------------");
//        System.out.println("\tTest Label: " + testLabel);
//        System.out.println("-----------------------------\n");

        for (int testSample = startTestIndex; testSample < endTestIndex; testSample++) {

            double[][] testRow = matrix.convertToRowMatrix(testFeatures, testSample);

            double prediction = matrix.dot(testRow, currentWeight)[0][0];
            int actualLabel = testLabels[testSample][0];
            
//            System.out.print((testSample + 1) + "\tActual: " + actualLabel + "\tPredicted: ");
            
            boolean positivePrediction = prediction >= 0;
            
            if (positivePrediction) {
                int predictedLabel = testLabel;
                boolean correctPrediction = predictedLabel == actualLabel;
//                System.out.print(predictedLabel);
                // total correct predictions
                if (correctPrediction) {
                    overallCorrect++;
                    currentLabelCorrect++;
                    // stores the prediction in the confidence matrix
                    predictionMatrix[testSample][testLabel] = prediction;
                } else {
//                    System.out.print(" <----------------- Misclassified (Overestimate)");
                    overestimate++;
                }
//                System.out.println();
            } else {
//                System.out.print("Other");
                
                if (testLabel == actualLabel) {
//                    System.out.print(" <----------------- Misclassified (Underestimate)");   
                    underestimate++;
                } else {
                    overallCorrect++;
                    // stores the prediction in the confidence matrix
                    predictionMatrix[testSample][testLabel] = prediction;
                }
//                System.out.println();
            }
            
            // number of test labels in the dataset
            if (testLabel == actualLabel) {
                totalTestLabels++;
            }
            
            // total number of samples in the dataset
            totalTestSamplesInLoop++;
        }
        
        // accuracy over current label
        totalError = totalTestLabels - currentLabelCorrect;
        accuracy = currentLabelCorrect / (double)totalTestLabels * 100;
        
        // accuracy over all the labels in the dataset
        double overallError = totalTestSamplesInLoop - (overestimate + underestimate);
        double overallAccuracy = overallError / totalTestSamplesInLoop * 100d;

        if (printTrainingResultsForLabel) {
            System.out.println("\n----------------------------------");
            System.out.println("\tTest Label: " + testLabel);
            System.out.println("----------------------------------");
            System.out.println("Total Label " + testLabel + ":\t\t" + totalTestLabels);
            System.out.println("Total Correct:\t\t" + currentLabelCorrect);
            System.out.println("Total Error:\t\t" + totalError);
            System.out.printf("Accuracy:\t\t%.2f", accuracy);
            System.out.println(" %");

            System.out.println("\nTotal Test Samples:\t" + totalTestSamplesInLoop);
            System.out.println("Overestimated:\t\t" + overestimate);
            System.out.println("Underestimated:\t\t" + underestimate);
            System.out.printf("Overall Accuracy:\t%.2f", overallAccuracy);
            System.out.println(" %");
            System.out.println("----------------------------------\n");
        }   
        
//            System.out.print(learningRate + "\t" + totalEpochs + "\t");
//            System.out.printf("%.2f\t%.2f\n", accuracy, overallAccuracy);
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
     * Adds a bias term of -1 at the end of the feature array of each sample
     */
    private void addBiasAtTheEnd() {
        for (int sample = 0; sample < totalTrainSamples; sample++) {
            trainFeatures[sample][totalTrainFeatures - 1] = -1;
        }
        
        for (int sample = 0; sample < totalTestSamples; sample++) {
            testFeatures[sample][totalTestFeatures - 1] = -1;
        }
    }
    
    /**
     * converts the current training labels to +1 and others to -1
     * @param label 
     */
    private void oneHotEncodeTrainLabels(int label) {
        
        // prepares the known output according to the label, if label = 0, corresponding output = 1, others = -1
        for (int sample = startTrainIndex; sample < endTrainIndex; sample++) {
            if (Double.compare(trainLabels[sample][0], label) == 0) {
                oneHotLabelsTemp[0][sample] = 1;
            } else {
                oneHotLabelsTemp[0][sample] = -1;
            }
        }
    }
    
    /**
     * takes the highest confidence value for each label, predicts it to be of that label class, stores the label value in the predictedLabel array
     */
    private void buildDecisionMatrix() {
        for (int sample = startTestIndex; sample < endTestIndex; sample++) {
            int predictedLabel = -1;
            double confidence = NEGATIVE_INFINITY;
            for (int label = 0; label < TOTAL_LABEL_CLASSES; label++) {
                if (predictionMatrix[sample][label] > confidence) {
                    confidence = predictionMatrix[sample][label];
                    predictedLabel = label;
                }
            }
            // store the label with the highest confidence in the classified matrix
            predictedMatrix[sample][0] = predictedLabel;
        }
    }
    
    /**
     * Builds the confusionMatrix and confusionMatrixErrorArray variables
     */
    private void buildConfusionMatrix() {
        
        // initializes all the indeices to 0
        for (int row = 0; row < TOTAL_LABEL_CLASSES; row++) {
            for (int col = 0; col < TOTAL_LABEL_CLASSES; col++) {
                confusionMatrix[row][col] = 0;
            }
        }
        
        // Error values for each label are stored in columns
        for (int sample = startTestIndex; sample < endTestIndex; sample++) {
            int actualLabel = testLabels[sample][0];
            int predictedLabel = predictedMatrix[sample][0];
            
            confusionMatrix[predictedLabel][actualLabel] += 1;
        }
        
        // stores the errors for each label class
        for (int col = 0; col < TOTAL_LABEL_CLASSES; col++) {
            for (int row = 0; row < TOTAL_LABEL_CLASSES; row++) {
                if (col != row) {
                    confusionMatrixErrorArray[col] += confusionMatrix[row][col];
                }
            }
        }
    }
    
    /**
     * save the trained weights
     * @param label 
     */
    private void saveWeights(int label) {
        // write the weights to a file
        if (SAVE_WEIGHTS) {
            try {
                FileScanner fileScanner = new FileScanner();
                fileScanner.saveWeights(label, tempWeight);
//                System.out.println("Weights for label " + label + " trained for " + TOTAL_EPOCHS + " epochs");

            } catch (IOException exception) {
                System.out.println(exception);
            }
        }
    }
    
    /**
     * load pretrained weights
     * @param label
     * @return 
     */
    private double[][] loadWeights(int label) {
        double[][] currentWeight = null;

        // loads pre-trained weights
        if (LOAD_WEIGHTS) {
            System.out.println("Loading pre-trained weights for label: " + label);
            FileScanner fileScanner = new FileScanner();
            currentWeight = fileScanner.loadWeights(label);
            System.out.println("Successfully loaded weights for label: " + label);

        } else {
            // if LOAD_WEIGHTS is set to false
            if (weightArray[label] == null) {
                System.out.println("Please train SVM for label: " + label);
            } else {
                currentWeight = weightArray[label];
            }
        }
        
        // if loaded weight not found and testWeight is null
        if ((currentWeight == null) && (LOAD_WEIGHTS)) {
            System.out.println("\nError loading weights for " + label);
            return null;
        }
        
        // if LOAD_WEIGHTS is false and testWeight is null
        if ((currentWeight == null) && (!LOAD_WEIGHTS)) {
            System.out.println("\nPlease train SVM for " + label + " before running!");
            return null;
        }
        
        return currentWeight;
    }
    
    /**
     * prints the initial parameters
     */
    private void printInitialParameters() {
        System.out.println("Learning Rate:\t" + learningRate);
        System.out.println("Total Epochs:\t" + totalEpochs);
        System.out.println("Load Weights:\t" + LOAD_WEIGHTS);
        System.out.println("Save Weights:\t" + SAVE_WEIGHTS);
        System.out.println("-----------------------------------------------\n\n");
    }
    
    /**
     * prints the weight vector
     * @param weightMatrix 
     */
    private void printWeights(double[][] weightMatrix) {
        // print the trained weights
        for (int w = 0; w < totalTrainFeatures; w++) {
            System.out.printf("\nWeight" + w + ": %f", weightMatrix[w][0]);
        }
    }
    
    /**
     * prints the confidence matrix (each row contains the confidence value for each label)
     */
    private void printConfidenceMatrix() {
        
        System.out.println("\n\n--------------------------------");
        System.out.println("\tConfidence Matrix");
        System.out.println("----------------------------------");
        for (int label = 0; label < 10; label++) {
            System.out.print("\t\t" + label);
        }
        System.out.println();
        for (int sample = 0; sample < totalTrainSamples; sample++) {
            System.out.print((sample+1) + ":");
            for (int label = 0; label < TOTAL_LABEL_CLASSES; label++) {
                if (predictionMatrix[sample][label] > 0) {
                    System.out.printf("\t\t+%.2f", predictionMatrix[sample][label]);
                } else {
                    System.out.printf("\t\t%.2f", predictionMatrix[sample][label]);
                }
            }
            System.out.println();
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
        
        System.out.println("\n-------------------------------------------------------------------------------------------");

        for (int row = 0; row < TOTAL_LABEL_CLASSES; row++) {
            System.out.print(row + " ->\t");

            for (int col = 0; col < TOTAL_LABEL_CLASSES; col++) {
                System.out.print(confusionMatrix[row][col] + "\t");
            }
            System.out.println("\n");
        }
        System.out.println("-------------------------------------------------------------------------------------------");
        
        System.out.print("Error->");
        for (int errorOnLabel = 0; errorOnLabel < TOTAL_LABEL_CLASSES; errorOnLabel++) {
            System.out.print("\t" + confusionMatrixErrorArray[errorOnLabel]);
        }
        System.out.println("\n");
    }
    
    /**
     * Prints the decission matrix
     */
    private void printDecisionMatrix() {
        
        System.out.println("\n\n--------------------------------");
        System.out.println("\tDecision Matrix");
        System.out.println("----------------------------------");
        
        for (int sample = 0; sample < totalTrainSamples; sample++) {
            System.out.println((sample+1) + ":\t" + predictedMatrix[sample][0]);
        }
    }
    
    /**
     * prints the true labels against the predicted labels
     */
    private void printFinalClassification() {
        System.out.println("\n\n----------------------------------------");
        System.out.println("\tFinal Classification Matrix");
        System.out.println("----------------------------------------");

        for (int sample = 0; sample < totalTrainSamples; sample++) {
            int actualLabel = testLabels[sample][0];
            int predictedLabel = predictedMatrix[sample][0];
            
//            System.out.print((sample+1));
//            System.out.print("\tActual:\t" + actualLabel);
//            System.out.print("\t|||\tPredicted: " + predictedLabel);
            
            if (actualLabel != predictedLabel) {
                System.out.print((sample+1));
                System.out.print("\tActual:\t" + actualLabel);
                System.out.print("\t|||\tPredicted: " + predictedLabel);
                
                System.out.print(" <----------------- Misclassified\n");
            }
//            System.out.println();
        }
    }
    
    /**
     * prints the final accuracy of the test
     */
    private void printFinalAccuracy() {
        int correctPrediction = 0;
        for (int sample = 0; sample < totalTrainSamples; sample++) {
            int actualLabel = testLabels[sample][0];
            int predictedLabel = predictedMatrix[sample][0];
            if (actualLabel == predictedLabel) {
                correctPrediction++;
            }
        }
        
        double accuracy = ((double)correctPrediction / totalTrainSamples) * 100;
        System.out.println("\n\n-------------------------------");
        if (testing) {
            System.out.println("\tTest Results");
        } else {
            System.out.println("\tValidation Results");
        }
        
        System.out.println("-------------------------------");
        System.out.println("Total Samples:\t\t" + totalTrainSamples);
        System.out.println("Correct Predictions:\t" + correctPrediction);
        System.out.println("Incorrect Predictions:\t" + (totalTrainSamples - correctPrediction));
        System.out.printf("Accuracy:\t\t%.2f", accuracy);
        System.out.println(" %");
        System.out.println("-------------------------------\n\n\n\n\n");
        if (testing) {
            testResults[currentFold] = accuracy;
        }
    }
}
