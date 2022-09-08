package org.learn.algorithm.leetcode;

import org.apache.tomcat.Jar;
import org.springframework.boot.context.properties.bind.handler.IgnoreTopLevelConverterNotFoundBindHandler;
import org.springframework.jdbc.core.metadata.TableMetaDataProvider;

public class VipDynamicProgram {

    public static void main(String[] args) {
        int[][] costs = new int[][]{{14, 2, 11}, {11, 14, 5}, {14, 3, 10}};

        VipDynamicProgram dynamicProgram = new VipDynamicProgram();

//        System.out.println(dynamicProgram.minCostIIFollowUp(costs));

//        dynamicProgram.numSquares(12);

        costs = new int[][]{{3, 5, 3}, {6, 17, 6}, {7, 13, 18}, {9, 10, 18}};

        dynamicProgram.minCostIIFollowUp(costs);


    }

    // 房屋染色系列 //

    /**
     * @param costs: n x 3 cost matrix
     * @return: An integer, the minimum cost to paint all houses
     */
    public int minCost(int[][] costs) {
        // write your code here
        if (costs == null || costs.length == 0) {
            return 0;
        }
        for (int i = 1; i < costs.length; i++) {
            int prev = i - 1;
            costs[i][0] += Math.min(costs[prev][1], costs[prev][2]);
            costs[i][1] += Math.min(costs[prev][0], costs[prev][2]);
            costs[i][2] += Math.min(costs[prev][0], costs[prev][1]);
        }
        int lastRow = costs.length - 1;
        return Math.min(Math.min(costs[lastRow][0], costs[lastRow][1]), costs[lastRow][2]);
    }

    /**
     * 265. Paint House II
     *
     * @param costs: n x k cost matrix
     * @return: an integer, the minimum cost to paint all houses
     */
    public int minCostII(int[][] costs) {
        // write your code here
        if (costs == null || costs.length == 0) {
            return 0;
        }
        int row = costs.length;
        int column = costs[0].length;
        for (int i = 1; i < costs.length; i++) {
            for (int j = 0; j < column; j++) {
                int val = Integer.MAX_VALUE;
                for (int k = 0; k < column; k++) {
                    if (k == j) {
                        continue;
                    }
                    if (costs[i - 1][k] < val) {
                        val = costs[i - 1][k];
                    }
                }
                costs[i][j] += val;
            }
        }
        int result = Integer.MAX_VALUE;
        for (int j = 0; j < column; j++) {
            result = Math.min(result, costs[row - 1][j]);
        }
        return result;
    }

    // todo
    public int minCostIIFollowUp(int[][] costs) {
        if (costs == null || costs.length == 0) {
            return 0;
        }
        int row = costs.length;
        int column = costs[0].length;

        int firstIndex = -1;
        int secondIndex = -1;
        for (int i = 0; i < row; i++) {

            int tmpFirstIndex = -1;
            int tmpSecondIndex = -1;

            for (int j = 0; j < column; j++) {
                int val = costs[i][j];

                if (firstIndex != -1 && j != firstIndex) {
                    costs[i][j] += costs[i - 1][firstIndex];
//                    System.out.println("current row: " + i + " first val:" + costs[i][j]);
                } else if (secondIndex != -1 && j != secondIndex) {
                    costs[i][j] += costs[i - 1][secondIndex];
//                    System.out.println("current row: " + i + " second val:" + costs[i][j]);
                }

                if (tmpFirstIndex == -1 || costs[i][tmpFirstIndex] > val) {
                    tmpSecondIndex = tmpFirstIndex;
                    tmpFirstIndex = j;
                } else if (tmpSecondIndex == -1 || costs[i][tmpSecondIndex] > val) {
                    tmpSecondIndex = j;
                }
            }
//            System.out.println("current row:" + i + " tmpFirst: " + tmpFirstIndex + " tmpSecond:" + tmpSecondIndex);
//            for (int j = 0; j < column; j++) {
//
//            }
            firstIndex = tmpFirstIndex;
            secondIndex = tmpSecondIndex;
        }
        return costs[row - 1][firstIndex];
//        return Math.min(costs[row - 1][firstIndex], costs[row - 1][secondIndex]);
    }


    /**
     * https://www.lintcode.com/problem/514/solution/25649
     * todo
     * 276 Paint Fence
     *
     * @param n: non-negative integer, n posts
     * @param k: non-negative integer, k colors
     * @return: an integer, the total number of ways
     */
    public int numWays(int n, int k) {
        // write your code here
        int[][] dp = new int[n + 1][k + 1];
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j < k; j++) {
                dp[i][j] = dp[i][j - 1]

            }
        }
        return 0;
    }


    /**
     * dp[i] = dp[i-j * j] + dp[j*j], dp[i-1];
     * 279. Perfect Squares
     *
     * @param n
     * @return
     */
    public int numSquares(int n) {
        if (n <= 1) {
            return n;
        }
        int[] dp = new int[n + 1];

        dp[1] = 1;

        for (int i = 1; i <= n; i++) {
            dp[i] = i;
            for (int j = 1; j * j <= i; j++) {
                if (j * j == i) {
                    dp[i] = 1;
                    break;
                } else {
                    int remain = i - j * j;
                    dp[i] = Math.min(dp[remain] + 1, dp[i]);
                }
            }
        }
        return dp[n];
    }


}
