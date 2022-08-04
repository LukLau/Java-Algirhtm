package org.learn.algorithm.leetcode;

public class VipDynamicProgram {

    public static void main(String[] args) {
        int[][] costs = new int[][]{{14, 2, 11}, {11, 14, 5}, {14, 3, 10}};

        VipDynamicProgram dynamicProgram = new VipDynamicProgram();

        System.out.println(dynamicProgram.minCostIIFollowUp(costs));

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
            costs[i][0] += Math.min(costs[i - 1][1], costs[i - 1][2]);
            costs[i][1] += Math.min(costs[i - 1][0], costs[i - 1][2]);
            costs[i][2] += Math.min(costs[i - 1][1], costs[i - 1][0]);
        }
        return Math.min(costs[costs.length - 1][0], Math.min(costs[costs.length - 1][2], costs[costs.length - 1][1]));
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

    public int minCostIIFollowUp(int[][] costs) {
        if (costs == null || costs.length == 0) {
            return 0;
        }
//        int row = costs.length;
        int column = costs[0].length;


        int firstSmall = -1;
        int secondSmall = -1;
        for (int i = 0; i < costs.length; i++) {
            int tmpFirst = -1;
            int tmpSecond = -1;

            for (int j = 0; j < column; j++) {

                if (firstSmall != -1 && j != firstSmall) {
                    costs[i][j] += costs[i - 1][firstSmall];
                } else if (secondSmall != -1 && j != secondSmall) {
                    costs[i][j] += costs[i - 1][secondSmall];
                }
                if (tmpFirst == -1 || costs[i][j] < costs[i][tmpFirst]) {
                    tmpSecond = tmpFirst;
                    tmpFirst = j;
                } else if (tmpSecond == -1 || costs[i][j] < costs[i][tmpSecond]) {
                    tmpSecond = j;
                }
            }
            firstSmall = tmpFirst;
            secondSmall = tmpSecond;
        }
        return costs[costs.length - 1][firstSmall];
    }

}
