package org.dora.algorithm.solution;

/**
 * 常见算法
 *
 * @author liulu
 * @date 2019-03-20
 */
public class InterviewAlgorithm {

    /**
     * 0-1背包问题
     *
     * @param weights
     * @param w
     * @param values
     * @param v
     * @return
     */
    public int knapsack(int[] weights, int w, int[] values, int v) {
        if (weights == null || values == null) {
            return 0;
        }
        int[][] dp = new int[v + 1][w + 1];
        for (int i = 1; i <= v; i++) {
            for (int j = 1; j <= w; j++) {
                if (weights[i - 1] > j) {
                    dp[i][j] = dp[i - 1][j];
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i - 1][j - weights[i - 1]] + values[i - 1]);
                }
            }
        }
        return dp[v][w];
    }
}
