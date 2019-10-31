/**
 * Class containing basic matrix generation and manipulation functions
 * @author Abhijit Adhikary
 */
class Matrix {
    
    /**
     * Default constructor of Matrix
     */
    public Matrix() {
    }
    
    /**
     * Generates a matrix of specified dimentions
     * @param numRows
     * @param numCol
     * @return 
     */
    public double[][] generateMatix(int numRows, int numCol) {
        if (numRows <= 0 || numCol <= 0) {
            System.out.println("Invalid matrix dimention");
            return null;
        }
        double[][] matrix = new double[numRows][numCol];
        
        for (int row = 0; row < numRows; row++) {
            for (int col = 0; col < numCol; col++) {
                matrix[row][col] = generateRandomValue(0, 5);
            }
        }
        
        return matrix;
    }
    
    /**
     * Returns a random value between specified intervals
     * @param min
     * @param max
     * @return 
     */
    private double generateRandomValue(int min, int max) {
        return ((Math.random() * max) + min);
    }
    
    /**
     * Prints a matrix
     * @param matrix 
     */
    public void printMatrix(double[][] matrix) {
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                System.out.print(matrix[i][j] + "\t");
            }
            System.out.println();
        }
        System.out.println();
    }
    
    /**
     * Returns the dot product of two matrices
     * @param firstMatrix
     * @param secondMatrix
     * @return 
     */
    public double[][] dot(double[][] firstMatrix, double[][] secondMatrix) {
        int row1 = firstMatrix.length;
        int column1 = firstMatrix[0].length;

        int row2 = secondMatrix.length;
        int column2 = secondMatrix[0].length;
        
        double[][] product;
        
        // if dimentions match up then computes the dot product
        if (column1 == row2) {
            product = new double[row1][column2];
            
            for(int row = 0; row < row1; row++) {
                for (int col = 0; col < column2; col++) {
                    for (int k = 0; k < column1; k++) {
                        product[row][col] += firstMatrix[row][k] * secondMatrix[k][col];
                    }
                }
            }

            return product;
        } else {
            System.out.println("Invalid matrix dimentions");
            return null;
        }
    }
    
    /**
     * Returns a transpose of a supplied matrix
     * @param matrix
     * @return 
     */
    public double[][] transpose(double[][] matrix) {
        int numRow = matrix.length;
        int numCol = matrix[0].length;
        double[][] transposedMatrix = new double[numCol][numRow];
        
        for (int row = 0; row < numRow; row++) {
            for (int col = 0; col < numCol; col++) {
                transposedMatrix[col][row] = matrix[row][col];
            }
        }
        return transposedMatrix;
    }
    
    /**
     * Returns a column matrix of specified length filled with zeros
     * @param length
     * @return 
     */
    public double[][] generateZeros(int length) {
        double[][] zeros = new double[length][1];
        
        for (int row = 0; row < length; row++) {
            zeros[row][0] = 0;
        }
        return zeros;
    }
    
    /**
     * Returns a matrix of specified length and width filled with zeros
     * @param length
     * @param width
     * @return 
     */
    public double[][] generateZeros(int length, int width) {
        double[][] zeros = new double[length][width];
        for (int row = 0; row < length; row++) {
            for (int col = 0; col < width; col++) {
                zeros[row][col] = 0;
            }
            
        }
        return zeros;
    }
    
    /**
     * Prints the shape of a supplied matrix
     * @param matrix 
     */
    public void printShape(double[][] matrix) {
        int row = matrix.length;
        int column = matrix[0].length;
        
        System.out.println("(" + row + ", " + column + ")");
    }
    
    /**
     * Returns a row matrix of a supplied matrix
     * @param matrix
     * @param row
     * @return 
     */
    public double[][] convertToRowMatrix(double[][] matrix, int row) {
        double rowMatrix[][] = new double[1][matrix[row].length];
        
        for (int column = 0; column < matrix[row].length; column++) {
            rowMatrix[0][column] = matrix[row][column];
        }
        
        return rowMatrix;
    }
    
    /**
     * Returns a matrix which is element wise multiplied with a scalar value
     * @param matrix
     * @param scalar
     * @return 
     */
    public double[][] elementwiseMultiply(double[][] matrix, double scalar) {
        double[][] newMatrix = new double[matrix.length][matrix[0].length];
        
        for (int row = 0; row < matrix.length; row++) {
            for (int column = 0; column < matrix[row].length; column++) {
                newMatrix[row][column] = scalar * matrix[row][column];
            }
        }
        return newMatrix;
    }
    
    /**
     * Returns a matrix which is element wise added with a scalar value
     * @param matrix
     * @param scalar
     * @return 
     */
    public double[][] elementwiseAdd(double[][] matrix, double scalar) {
        double[][] newMatrix = new double[matrix.length][matrix[0].length];
        for (int row = 0; row < matrix.length; row++) {
            for (int column = 0; column < matrix[row].length; column++) {
                newMatrix[row][column] = scalar + matrix[row][column];
            }
        }
        return newMatrix;
    }
    
    /**
     * Returns a matrix which is element wise added with another matrix
     * @param matrixA
     * @param matrixB
     * @return 
     */
    public double[][] elementwiseAdd(double[][] matrixA, double[][] matrixB) {
        double[][] newMatrix = new double[matrixA.length][matrixA[0].length]; 
        
        for (int row = 0; row < matrixA.length; row++) {
            for (int column = 0; column < matrixA[row].length; column++) {
                newMatrix[row][column] = matrixA[row][column] + matrixB[row][column];
            }
        }
        return newMatrix;
    }
    
}