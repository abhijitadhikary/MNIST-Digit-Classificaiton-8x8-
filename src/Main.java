/**
 * 
 * @author Abhijit Adhikary
 * To run nearest neighbor classifier run the runNearestNeighbor() method.
 * To soft margin support vector machine classifier run the runSVM() method.
 * Both algorighms run a 2 fold test by default
 * To change datasets rename the DATASET_1_FILENAME and DATASET_2_FILENAME variables
 */
class Main {
    private static final String DATASET_1_FILENAME = "cw2DataSet1.csv";
    private static final String DATASET_2_FILENAME = "cw2DataSet2.csv";  
    
    
    public static void main(String[] args) {
        runNearestNeighbor();
        runSVM();
    }
    
    /**
     * method for running the nearest neighbor classifier
     */
    private static void runNearestNeighbor() {
        NearestNeighbor nn = new NearestNeighbor(DATASET_1_FILENAME, DATASET_2_FILENAME);
        nn.run();
    }
    
    /**
     * method for running support vector machine
     */
    private static void runSVM() {
        SVM svm = new SVM(DATASET_1_FILENAME, DATASET_2_FILENAME);
        svm.runAllFolds();
    }
   
    
}