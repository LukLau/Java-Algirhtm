package org.learn.algorithm.datastructure;

/**
 * https://leetcode.com/problems/range-sum-query-2d-immutable/solution/
 * @author luk
 * @date 2021/4/27
 */
public class NumMatrix {
    private int[][] matrix;

    public NumMatrix(int[][] matrix) {
        this.matrix = matrix;
    }

    public int sumRegion(int row1, int col1, int row2, int col2) {
        int sum = 0;
        for (int i = row1; i <= row2; i++) {
            for (int j = col1; j <= col2; j++) {
                sum += this.matrix[i][j];
            }
        }
        return sum;
    }
}
