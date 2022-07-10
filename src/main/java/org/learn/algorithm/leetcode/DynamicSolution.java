package org.learn.algorithm.leetcode;

import org.learn.algorithm.datastructure.TreeNode;

import java.util.*;

/**
 * 动态规划问题
 *
 * @author luk
 * @date 2021/4/8
 */
public class DynamicSolution {


    public static void main(String[] args) {
        DynamicSolution solution = new DynamicSolution();
        char[][] matrix = new char[][]{{'0', '1', '1', '0', '1'}, {'1', '1', '0', '1', '0'}, {'0', '1', '1', '1', '0'}, {'1', '1', '1', '1', '0'}, {'1', '1', '1', '1', '1'}, {'0', '0', '0', '0', '0'}};
//        solution.numDecodings("12");
//        solution.maximalRectangle(matrix);
        List<List<Integer>> lists = solution.generate(4);
        System.out.println(lists);

//        solution.isInterleave("", "", "");
//        char[][] nums = new char[][]{{'1', '0', '1', '0', '0'}, {'1', '0', '1', '1', '1'}, {"1", "1", "1", "1", "1"}, {"1", "0", "0", "1", "0"}}
//        solution.maximalRectangle(nums);
    }
    // 背包系列问题

    /**
     * 该系列主要分为三个
     * 0-1背包问题 Fn = max(Dp[n][w], Dp[n-1][n -w[i][j]] + v[j]);
     * 完全背包问题 Fn = max(Dp[n][w], Dp[n][n -w[i][j]] + v[j]);
     * 部分背包问题 贪心算法
     */


    // 0-1 背包问题


    // 普通动态规划问题

    /**
     * @param repository
     * @param customerQuery
     * @return
     */
    public static List<List<String>> searchSuggestions(List<String> repository, String customerQuery) {
        // Write your code here
        if (repository == null || repository.isEmpty()) {
            return new ArrayList<>();
        }
        if (customerQuery == null || customerQuery.isEmpty()) {
            return new ArrayList<>();
        }
        List<List<String>> result = new ArrayList<>();
        int m = customerQuery.length();
        for (int i = 1; i < m; i++) {
            List<String> tmp = new ArrayList<>();

            String prefix = customerQuery.substring(0, i + 1);

            for (String item : repository) {

                if (item.contains(prefix)) {
                    tmp.add(item);
                }
            }
            if (!tmp.isEmpty()) {
                result.add(tmp);
            }
        }
        return result;
    }


    // 序列问题

    /**
     * todo
     * 91. Decode Ways
     *
     * @param s
     * @return
     */
    public int numDecodings(String s) {
        if (s == null || s.length() == 0) {
            return 0;
        }
        int len = s.length();
        int[] dp = new int[len + 1];
        dp[0] = 1;
        for (int i = 1; i <= len; i++) {
            String first = s.substring(i - 1, i);
            int firstValue = Integer.parseInt(first);
            if (firstValue >= 1 && firstValue <= 9) {
                dp[i] += dp[i - 1];
            }
            String second = i >= 2 ? s.substring(i - 2, i) : "0";
            int secondValue = Integer.parseInt(second);
            if (secondValue >= 10 && secondValue <= 26) {
                dp[i] += dp[i - 2];
            }
        }
        return dp[len];
    }


    // 八皇后问题 //

    /**
     * 300. Longest Increasing Subsequence
     *
     * @param nums
     * @return
     */
    public int lengthOfLIS(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int[] dp = new int[nums.length];
        Arrays.fill(dp, 1);
        for (int i = 1; i < nums.length; i++) {
            for (int j = 0; j < i; j++) {
                if (dp[i] > dp[j] && dp[i] < dp[j] + 1) {
                    dp[i] = dp[j] + 1;
                }
            }
        }
        int result = 0;
        for (int tmp : dp) {
            result = Math.max(result, tmp);
        }
        return result;
    }

    /**
     * @param n
     * @return
     */
    public List<List<String>> solveNQueens(int n) {
        if (n <= 0) {
            return new ArrayList<>();
        }
        char[][] queen = new char[n][n];
        for (char[] row : queen) {
            Arrays.fill(row, '.');
        }
        List<List<String>> result = new ArrayList<>();

        intervalNQueens(result, 0, n, queen);
        return result;
    }

    private void intervalNQueens(List<List<String>> result, int row, int n, char[][] queens) {
        if (row == n) {
            List<String> tmp = new ArrayList<>();
            for (char[] word : queens) {
                tmp.add(String.valueOf(word));
            }
            result.add(tmp);
            return;
        }
        for (int i = 0; i < n; i++) {
            if (isValidQueens(queens, i, row, n)) {
                queens[row][i] = 'Q';
                intervalNQueens(result, row + 1, n, queens);
                queens[row][i] = '.';
            }
        }
    }

    private boolean isValidQueens(char[][] queens, int col, int row, int n) {
        for (int i = row - 1; i >= 0; i--) {
            if (queens[i][col] == 'Q') {
                return false;
            }
        }
        for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--) {
            if (queens[i][j] == 'Q') {
                return false;
            }
        }
        for (int i = row - 1, j = col + 1; i >= 0 && j < n; i--, j++) {
            if (queens[i][j] == 'Q') {
                return false;
            }
        }
        return true;
    }

    /**
     * 52. N-Queens II
     *
     * @param n
     * @return
     */
    public int totalNQueens(int n) {
        if (n <= 0) {
            return 0;
        }
        int[] dp = new int[n];
        return intervalTotalQueens(dp, 0, n);
    }

    private int intervalTotalQueens(int[] dp, int row, int n) {
        int count = 0;
        if (row == n) {
            return 1;
        }
        for (int i = 0; i < n; i++) {
            if (isValidTotalQueens(dp, i, row, n)) {
                dp[row] = i;
                count += intervalTotalQueens(dp, row + 1, n);
                dp[row] = -1;
            }
        }
        return count;
    }


    // ----- //

    // --编辑距离问题 //

    private boolean isValidTotalQueens(int[] dp, int col, int row, int n) {
        for (int i = row - 1; i >= 0; i--) {
            if (dp[i] == col || Math.abs(i - row) == Math.abs(dp[i] - col)) {
                return false;
            }
        }
        return true;
    }

    /**
     * 72. Edit Distance
     *
     * @param word1
     * @param word2
     * @return
     */
    public int minDistance(String word1, String word2) {
        if (word1 == null || word2 == null) {
            return 0;
        }
        int m = word1.length();
        int n = word2.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++) {
            dp[i][0] = i;
        }
        for (int j = 1; j <= n; j++) {
            dp[0][j] = j;
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = 1 + Math.min(dp[i][j - 1], Math.min(dp[i - 1][j - 1], dp[i - 1][j]));
                }
            }
        }
        return dp[m][n];
    }

    /**
     * todo
     * 85. Maximal Rectangle
     *
     * @param matrix
     * @return
     */
    public int maximalRectangle(char[][] matrix) {
        if (matrix == null || matrix.length == 0) {
            return 0;
        }
        int row = matrix.length;
        int column = matrix[0].length;
        int result = 0;
        int[] left = new int[column];
        int[] height = new int[column];
        int[] right = new int[column];
        Arrays.fill(right, column);
        for (int i = 0; i < row; i++) {
            int leftSide = 0;
            int rightSide = column;
            for (int j = 0; j < column; j++) {
                char word = matrix[i][j];
                if (word == '1') {
                    height[j] = height[j] + 1;
                    left[j] = Math.max(left[j], leftSide);
                } else {
                    height[j] = 0;
                    left[j] = leftSide;
                    leftSide = j + 1;
                }
            }
            for (int j = column - 1; j >= 0; j--) {
                char word = matrix[i][j];
                if (word == '1') {
                    right[j] = Math.min(right[j], rightSide);
                } else {
                    right[j] = column;
                    rightSide = j;
                }
            }
            for (int j = 0; j < column; j++) {
                if (height[j] == 0) {
                    continue;
                }
                int val = height[j] * (right[j] - left[j]);
                if (val >= result) {
                    System.out.println("row:" + i + " column: " + j + " result:" + val);
                    result = val;
                }
            }
        }
        // 0 1 1 3 4
        // 2 2 3 4 5
        // --------//
        // 0 1 1 3 0
        // 4 2 4 4 5
        return result;
    }

    /**
     * 221. Maximal Square
     *
     * @param matrix
     * @return
     */
    public int maximalSquare(char[][] matrix) {
        if (matrix == null || matrix.length == 0) {
            return 0;
        }
        int row = matrix.length;
        int column = matrix[0].length;
        int[][] dp = new int[row][column];

        int result = 0;
        for (int i = 0; i < row; i++) {
            if (matrix[i][0] == '1') {
                dp[i][0] = 1;
                result = 1;
            }
        }
        for (int j = 0; j < column; j++) {
            if (matrix[0][j] == '1') {
                dp[0][j] = 1;
                result = 1;
            }
        }
        for (int i = 1; i < row; i++) {
            for (int j = 1; j < column; j++) {
                if (matrix[i][j] == '1') {
                    dp[i][j] = 1 + Math.min(dp[i - 1][j], Math.min(dp[i][j - 1], dp[i - 1][j - 1]));
                    result = Math.max(result, dp[i][j] * dp[i][j]);
                }
            }
        }
        return result;
    }

    /**
     * 97. Interleaving String
     *
     * @param s1
     * @param s2
     * @param s3
     * @return
     */
    public boolean isInterleave(String s1, String s2, String s3) {
        if (s1 == null || s2 == null || s3 == null) {
            return false;
        }
        int m = s1.length();
        int n = s2.length();
        if (m + n != s3.length()) {
            return false;
        }
        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[0][0] = true;
        for (int i = 1; i <= m; i++) {
            dp[i][0] = s1.charAt(i - 1) == s3.charAt(i - 1) && dp[i - 1][0];
        }
        for (int j = 1; j <= n; j++) {
            dp[0][j] = s2.charAt(j - 1) == s3.charAt(j - 1) && dp[0][j - 1];
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                dp[i][j] = (dp[i - 1][j] && s1.charAt(i - 1) == s3.charAt(i + j - 1)) || (dp[i][j - 1] && s2.charAt(j - 1) == s3.charAt(i + j - 1));
            }
        }
        return dp[m][n];
    }

    /**
     * 115. Distinct Subsequences
     *
     * @param s
     * @param t
     * @return
     */
    public int numDistinct(String s, String t) {
        if (s == null || t == null) {
            return 0;
        }
        int m = s.length();
        int n = t.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++) {
            dp[i][0] = 1;
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                dp[i][j] = dp[i - 1][j] + (s.charAt(i - 1) == t.charAt(j - 1) ? dp[i - 1][j - 1] : 0);
            }
        }
        return dp[m][n];
    }

    /**
     * 118. Pascal's Triangle
     *
     * @param numRows
     * @return
     */
    public List<List<Integer>> generate(int numRows) {
        if (numRows < 0) {
            return new ArrayList<>();
        }
        List<List<Integer>> result = new ArrayList<>();
        for (int i = 0; i < numRows; i++) {
            List<Integer> tmp = new ArrayList<>();
            tmp.add(1);
            for (int j = 1; j <= i - 1; j++) {
                List<Integer> previous = result.get(i - 1);
                int val = previous.get(j - 1) + previous.get(j);
                tmp.add(val);
            }
            if (i > 0) {
                tmp.add(1);
            }
            result.add(tmp);
        }
        return result;
    }

    /**
     * 119. Pascal's Triangle II
     *
     * @param rowIndex
     * @return
     */
    public List<Integer> getRow(int rowIndex) {
        if (rowIndex < 0) {
            return new ArrayList<>();
        }
        List<Integer> result = new ArrayList<>();

        result.add(1);

        for (int i = 0; i <= rowIndex; i++) {

            for (int j = i - 1; j >= 1; j--) {
                int val = result.get(j) + result.get(j - 1);
                result.set(j, val);
            }
            if (i > 0) {
                result.add(1);
            }
        }
        return result;
    }


    // -卖股票系列问题- //

    /**
     * 120. Triangle
     *
     * @param triangle
     * @return
     */
    public int minimumTotal(List<List<Integer>> triangle) {
        if (triangle == null || triangle.isEmpty()) {
            return Integer.MAX_VALUE;
        }
        int size = triangle.size();
        for (int i = size - 2; i >= 0; i--) {
            int currentLevel = triangle.get(i).size();
            for (int j = 0; j < currentLevel; j++) {
                int val = Math.min(triangle.get(i + 1).get(j), triangle.get(i + 1).get(j + 1)) + triangle.get(i).get(j);
                triangle.get(i).set(j, val);
            }
        }
        return triangle.get(0).get(0);
    }

    /**
     * 135. Candy
     *
     * @param ratings
     * @return
     */
    public int candy(int[] ratings) {
        if (ratings == null || ratings.length == 0) {
            return 0;
        }
        int[] dp = new int[ratings.length];
        Arrays.fill(dp, 1);
        for (int i = 1; i < ratings.length; i++) {
            if (ratings[i] > ratings[i - 1] && dp[i] < dp[i - 1] + 1) {
                dp[i] = dp[i - 1] + 1;
            }
        }
        for (int i = ratings.length - 2; i >= 0; i--) {
            if (ratings[i] > ratings[i + 1] && dp[i] < dp[i + 1] + 1) {
                dp[i] = dp[i + 1] + 1;
            }
        }
        int result = 0;
        for (int count : dp) {
            result += count;
        }
        return result;
    }

    /**
     * 121. Best Time to Buy and Sell Stock
     *
     * @param prices
     * @return
     */
    public int maxProfit(int[] prices) {
        if (prices == null || prices.length == 0) {
            return 0;
        }
        int result = 0;
        int cost = prices[0];
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > cost) {
                result = Math.max(result, prices[i] - cost);
            } else {
                cost = prices[i];
            }
        }
        return result;
    }

    /**
     * 122. Best Time to Buy and Sell Stock II
     *
     * @param prices
     * @return
     */
    public int maxProfitII(int[] prices) {
        if (prices == null || prices.length == 0) {
            return 0;
        }
        int result = 0;
        int cost = prices[0];
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > cost) {
                result += prices[i] - cost;
            }
            cost = prices[i];
        }
        return result;
    }

    /**
     * 123. Best Time to Buy and Sell Stock III
     *
     * @param prices
     * @return
     */
    public int maxProfitIII(int[] prices) {
        if (prices == null || prices.length == 0) {
            return 0;
        }
        int len = prices.length;
        int[] left = new int[len];
        int leftResult = 0;
        int leftCost = prices[0];
        for (int i = 1; i < len; i++) {
            if (prices[i] > leftCost) {
                leftResult = Math.max(leftResult, prices[i] - leftCost);
            } else {
                leftCost = prices[i];
            }
            left[i] = leftResult;
        }
        int[] right = new int[len + 1];
        int rightResult = 0;
        int rightCost = prices[len - 1];
        for (int i = len - 2; i >= 0; i--) {
            if (prices[i] < rightCost) {
                rightResult = Math.max(rightResult, rightCost - prices[i]);
            } else {
                rightCost = prices[i];
            }
            right[i] = rightResult;
        }
        int result = 0;
        for (int i = 1; i < left.length; i++) {
            result = Math.max(result, left[i] + right[i]);
        }
        return result;
    }

    /**
     * todo
     * 188. Best Time to Buy and Sell Stock IV
     * 卖股票问题iv
     *
     * @param k
     * @param prices
     * @return
     */
    public int maxProfitIV(int k, int[] prices) {
        if (prices == null || prices.length == 0) {
            return 0;
        }
        int[][] dp = new int[k + 1][prices.length];
        for (int i = 1; i <= k; i++) {
            int cost = prices[0];
            for (int j = 1; j < prices.length; j++) {
                dp[i][j] = Math.max(dp[i][j - 1], prices[j] - cost);
                cost = Math.min(cost, prices[j] - dp[i - 1][j - 1]);
            }
        }
        return dp[k][prices.length - 1];
    }


    // 房屋抢劫系列

    /**
     * todo
     * 309. Best Time to Buy and Sell Stock with Cooldown
     *
     * @param prices
     * @return
     */
    public int maxProfitV(int[] prices) {
        int len = prices.length;
        int[] sell = new int[len];
        int[] buy = new int[len];
        buy[0] = -prices[0];
        for (int i = 1; i < len; i++) {
            sell[i] = Math.max(sell[i - 1], buy[i - 1] + prices[i]);
            buy[i] = Math.max(buy[i - 1], i == 1 ? -prices[i] : sell[i - 2] - prices[i]);
        }
        return sell[len - 1];
    }

    /**
     * 198. House Robber
     *
     * @param nums
     * @return
     */
    public int rob(int[] nums) {
        int robCurrent = 0;
        int robPrev = 0;
        for (int num : nums) {
            int tmp = robCurrent;
            robCurrent = Math.max(robCurrent, robPrev + num);
            robPrev = tmp;
        }
        return Math.max(robCurrent, robPrev);
    }

    /**
     * 213. House Robber II
     *
     * @param nums
     * @return
     */
    public int robII(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        if (nums.length == 1) {
            return 1;
        }
        return Math.max(intervalRob(nums, 1, nums.length - 1), intervalRob(nums, 0, nums.length - 2));
    }

    // 房子颜色问题

    private int intervalRob(int[] nums, int start, int end) {
        int robPrev = 0;
        int robCurrent = 0;
        for (int i = start; i <= end; i++) {
            int tmp = robCurrent;
            robCurrent = Math.max(robPrev + nums[i], robCurrent);
            robPrev = tmp;
        }
        return Math.max(robCurrent, robPrev);
    }

    /**
     * @param costs: n x 3 cost matrix
     * @return: An integer, the minimum cost to paint all houses
     */
    public int minCost(int[][] costs) {
        // write your code here
        if (costs == null || costs.length == 0) {
            return 0;
        }
        int row = costs.length;
        for (int i = 1; i < row; i++) {
            costs[i][0] += Math.min(costs[i - 1][1], costs[i - 1][2]);

            costs[i][1] += Math.min(costs[i - 1][0], costs[i - 1][2]);

            costs[i][2] += Math.min(costs[i - 1][1], costs[i - 1][0]);
        }
        return Math.min(Math.min(costs[row - 1][0], costs[row - 1][1]), costs[row - 1][2]);
    }


    // 普通题

    /**
     * O(n3) 的时间复杂度
     * 265 Paint House II
     * Hard
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
        for (int i = 1; i < row; i++) {
            for (int j = 0; j < column; j++) {
                int value = Integer.MAX_VALUE;
                for (int k = 0; k < column; k++) {
                    if (k == j) {
                        continue;
                    }
                    value = Math.min(value, costs[i - 1][k]);
                }
                costs[i][j] += value;
            }
        }
        int result = Integer.MAX_VALUE;
        for (int j = 0; j < column; j++) {
            result = Math.min(result, costs[row - 1][j]);
        }
        return result;
    }

    public int minCostIIV2(int[][] costs) {
        if (costs == null || costs.length == 0) {
            return 0;
        }
        int firstIndex = -1;
        int secondIndex = -1;
        int row = costs.length;
        int column = costs[0].length;
        for (int i = 0; i < row; i++) {
            int tmpFirstIndex = -1;
            int tmpSecondIndex = -1;
            for (int j = 0; j < column; j++) {
                if (j != firstIndex) {
                    costs[i][j] += firstIndex == -1 ? 0 : costs[i - 1][firstIndex];
                } else if (j != secondIndex) {
                    costs[i][j] += secondIndex == -1 ? 0 : costs[i - 1][secondIndex];
                }
                if (tmpFirstIndex == -1 || costs[i][j] < costs[i][tmpFirstIndex]) {
                    tmpSecondIndex = tmpFirstIndex;
                    tmpFirstIndex = j;
                } else if (tmpSecondIndex == -1 || costs[i][j] < costs[i][tmpSecondIndex]) {
                    tmpSecondIndex = j;
                }
            }
            firstIndex = tmpFirstIndex;
            secondIndex = tmpSecondIndex;
        }
        return costs[row - 1][firstIndex];
    }

    /**
     * 174. Dungeon Game
     *
     * @param dungeon
     * @return
     */
    public int calculateMinimumHP(int[][] dungeon) {
        if (dungeon == null || dungeon.length == 0) {
            return 0;
        }
        int row = dungeon.length;
        int column = dungeon[0].length;
        int[][] dp = new int[row][column];
        for (int j = column - 1; j >= 0; j--) {
            if (j == column - 1) {
                dp[row - 1][j] = Math.max(1, 1 - dungeon[row - 1][j]);
            } else {
                dp[row - 1][j] = Math.max(1, dp[row - 1][j + 1] - dungeon[row - 1][j]);
            }
        }
        for (int i = row - 2; i >= 0; i--) {
            for (int j = column - 1; j >= 0; j--) {
                if (j == column - 1) {
                    dp[i][j] = Math.max(1, dp[i + 1][j] - dungeon[i][j]);
                } else {
                    dp[i][j] = Math.max(1, Math.min(dp[i + 1][j], dp[i][j + 1]) - dungeon[i][j]);
                }
            }
        }
        return dp[0][0];
    }

    public int calculateMinimumHPII(int[][] dungeon) {
        int row = dungeon.length;
        int column = dungeon[0].length;
        int[] dp = new int[column];

        for (int j = column - 1; j >= 0; j--) {
            if (j == column - 1) {
                dp[j] = Math.max(1, 1 - dungeon[row - 1][j]);
            } else {
                dp[j] = Math.max(1, dp[j + 1] - dungeon[row - 1][j]);
            }
        }

        for (int i = row - 2; i >= 0; i--) {
            for (int j = column - 1; j >= 0; j--) {
                if (j == column - 1) {
                    dp[j] = Math.max(1, dp[j] - dungeon[i][j]);
                } else {
                    dp[j] = Math.max(1, Math.min(dp[j], dp[j + 1]) - dungeon[i][j]);
                }
            }
        }
        return dp[0];
    }


    /**
     * 289. Game of Life
     *
     * @param board
     */
    public void gameOfLife(int[][] board) {
        if (board == null || board.length == 0) {
            return;
        }
        int row = board.length;
        int column = board[0].length;
        int[][] matrix = new int[][]{{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                intervalGame(i, j, board, matrix);
            }
        }
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (board[i][j] == -1) {
                    board[i][j] = 0;
                } else if (board[i][j] == -2) {
                    board[i][j] = 1;
                }
            }
        }

    }

    private void intervalGame(int i, int j, int[][] board, int[][] matrix) {
        int liveCount = 0;
        for (int[] item : matrix) {
            int x = i + item[0];
            int y = j + item[1];
            if (x < 0 || x == board.length || y < 0 || y == board[x].length) {
                continue;
            }
            if (Math.abs(board[x][y]) == 1) {
                liveCount++;
            }
        }
        if (board[i][j] == 1 && !(liveCount == 2 || liveCount == 3)) {
            board[i][j] = -1;
        }
        if (board[i][j] == 0 && liveCount == 3) {
            board[i][j] = -2;
        }
    }

    /**
     * todo
     * 279. Perfect Squares
     *
     * @param n
     * @return
     */
    public int numSquares(int n) {
        if (n <= 0) {
            return 0;
        }
        int[] dp = new int[n + 1];
        dp[0] = 0;
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            int tmp = i;
            for (int j = 1; j * j <= i; j++) {
                tmp = Math.min(dp[i - j * j] + 1, tmp);
            }
            dp[i] = tmp;
        }
        return dp[n];
    }

    /**
     * todo
     * 312. Burst Balloons
     *
     * @param nums: A list of integer
     * @return: An integer, maximum coins
     */
    public int maxCoins(int[] nums) {
        // write your code here
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int len = nums.length;
        int[] dp = new int[len + 2];
        dp[0] = 1;
        dp[len + 1] = 1;
        for (int i = 1; i <= len; i++) {
        }
        for (int i = 1; i <= len; i++) {
            dp[i] = nums[i - 1];
        }
        return -1;
    }

    /**
     * 314
     * Binary Tree Vertical Order Traversal
     *
     * @param root
     * @return
     */
    public List<List<Integer>> verticalOrder(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        Queue<Integer> columnQueue = new LinkedList<>();
        Map<Integer, List<Integer>> map = new TreeMap<>();
        columnQueue.offer(0);
        Queue<TreeNode> nodeQueue = new LinkedList<>();
        nodeQueue.add(root);
        while (!nodeQueue.isEmpty()) {
            Integer poll = columnQueue.poll();
            TreeNode node = nodeQueue.poll();
            List<Integer> tmp = map.getOrDefault(poll, new ArrayList<>());
            tmp.add(node.val);
            map.put(poll, tmp);
            if (node.left != null) {
                columnQueue.offer(poll - 1);
                nodeQueue.offer(node.left);
            }
            if (node.right != null) {
                columnQueue.offer(poll + 1);
                nodeQueue.offer(node.right);
            }
        }
        return new ArrayList<>(map.values());
    }

    /**
     * NC138 矩阵最长递增路径
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * 递增路径的最大长度
     *
     * @param matrix int整型二维数组 描述矩阵的每个数
     * @return int整型
     */
    public int incrementPath(int[][] matrix) {
        // write code here
        if (matrix == null || matrix.length == 0) {
            return 0;
        }
        int row = matrix.length;
        int column = matrix[0].length;
        int result = 0;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                int tmp = maxPath(0, i, j, matrix, Integer.MIN_VALUE);
                result = Math.max(result, tmp);
            }
        }
        return result;
    }

    private int maxPath(int count, int i, int j, int[][] matrix, int minValue) {
        if (i < 0 || i == matrix.length || j < 0 || j == matrix[i].length) {
            return count;
        }
        // 如果不符合递增规律
        if (matrix[i][j] <= minValue) {
            return count;
        }
        // 符合 计数 + 1
        count++;

        int val = matrix[i][j];

        // 向上
        int top = Math.max(count, maxPath(count, i - 1, j, matrix, val));

        // 向下
        int bottom = Math.max(count, maxPath(count, i + 1, j, matrix, val));

        // 向左
        int left = Math.max(count, maxPath(count, i, j - 1, matrix, val));

        // 向右
        int right = Math.max(count, maxPath(count, i, j + 1, matrix, val));

        return Math.max(Math.max(top, bottom), Math.max(left, right));
    }
}
