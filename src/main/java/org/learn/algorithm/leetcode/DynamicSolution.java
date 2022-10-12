package org.learn.algorithm.leetcode;

import org.learn.algorithm.datastructure.TreeNode;
import org.springframework.boot.context.properties.bind.handler.IgnoreTopLevelConverterNotFoundBindHandler;

import java.awt.image.Kernel;
import java.util.*;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.locks.ReentrantLock;

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
//        List<List<Integer>> lists = solution.generate(4);
//        System.out.println(lists);

//        solution.isInterleave("", "", "");
//        char[][] nums = new char[][]{{'1', '0', '1', '0', '0'}, {'1', '0', '1', '1', '1'}, {"1", "1", "1", "1", "1"}, {"1", "0", "0", "1", "0"}}
//        solution.maximalRectangle(nums);

//        int[][] games = new int[][]{{0, 1, 0}, {0, 0, 1}, {1, 1, 1}, {0, 0, 0}};
//        int[][] games = new int[][]{{0, 0}};

//        solution.gameOfLife(games);
//        solution.numDecodings("226");
//        System.out.println(solution.generate(5));
//        solution.getRow(3);
//        solution.minCut("aab");
        matrix = new char[][]{{'1', '0'}};
//        solution.maximalSquare(matrix);
        solution.numSquares(4);
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
        if (s == null || s.isEmpty()) {
            return 0;
        }
        int m = s.length();
        int[] dp = new int[m + 1];

        dp[0] = 1;
        for (int i = 1; i <= m; i++) {
            String firstValue = s.substring(i - 1, i);

            int firstNum = Integer.parseInt(firstValue);

            if (firstNum >= 1 && firstNum <= 9) {
                dp[i] += dp[i - 1];
            }
            String secondValue = i >= 2 ? s.substring(i - 2, i) : "0";
            int secondNum = Integer.parseInt(secondValue);
            if (secondNum >= 10 && secondNum <= 26) {
                dp[i] += dp[i - 2];
            }
        }
        return dp[m];
    }


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
        int[] result = new int[nums.length];
        Arrays.fill(result, 1);
        for (int i = 1; i < nums.length; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[j] < nums[i] && result[i] < result[j] + 1) {
                    result[i] = result[j] + 1;
                }
            }
        }
        int answer = 0;
        for (int num : result) {
            answer = Math.max(answer, num);
        }
        return answer;
    }


    // 八皇后问题 //

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
        return internalTotalNQueens(dp, 0, n);
    }

    private int internalTotalNQueens(int[] dp, int row, int n) {
        if (row == n) {
            return 1;
        }
        int count = 0;
        for (int j = 0; j < n; j++) {
            if (isValidTotalQueens(dp, j, row, n)) {
                dp[row] = j;
                count += internalTotalNQueens(dp, row + 1, n);
                dp[row] = -1;
            }
        }
        return count;
    }

    private boolean isValidTotalQueens(int[] dp, int col, int row, int n) {
        for (int i = row - 1; i >= 0; i--) {
            if (dp[i] == col || Math.abs(i - row) == Math.abs(dp[i] - col)) {
                return false;
            }
        }
        return true;
    }


    /**
     * 63. Unique Paths II
     *
     * @param obstacleGrid
     * @return
     */
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        if (obstacleGrid == null || obstacleGrid.length == 0) {
            return 0;
        }
        int row = obstacleGrid.length;

        int column = obstacleGrid[0].length;

        int[] result = new int[column];
        for (int j = 0; j < column; j++) {
            if (obstacleGrid[0][j] == 1) {
                result[j] = 0;
            } else {
                result[j] = j == 0 ? 1 : result[j - 1];
            }
        }
        for (int i = 1; i < row; i++) {
            for (int j = 0; j < column; j++) {
                int val = obstacleGrid[i][j];
                if (val == 1) {
                    result[j] = 0;
                } else if (j > 0) {
                    result[j] += result[j - 1];
                }
            }
        }
        return result[column - 1];
    }


    /**
     * 64. Minimum Path Sum
     *
     * @param grid
     * @return
     */
    public int minPathSum(int[][] grid) {
        if (grid == null || grid.length == 0) {
            return 0;
        }
        int row = grid.length;
        int column = grid[0].length;
        int[] dp = new int[column];
        for (int j = 0; j < column; j++) {
            dp[j] = grid[0][j] + (j == 0 ? 0 : dp[j - 1]);
        }
        for (int i = 1; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (j == 0) {
                    dp[j] += grid[i][j];
                } else {
                    dp[j] = Math.min(dp[j], dp[j - 1]) + grid[i][j];
                }
            }
        }
        return dp[column - 1];
    }


    // ----- //

    // --编辑距离问题 //

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
                    dp[i][j] = Math.min(Math.min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]) + 1;
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
                char tmp = matrix[i][j];
                if (tmp == '1') {
                    height[j]++;
                    left[j] = Math.max(left[j], leftSide);
                } else {
                    height[j] = 0;
                    left[j] = leftSide;
                    leftSide = j + 1;
                }
            }
            for (int j = column - 1; j >= 0; j--) {
                char tmp = matrix[i][j];
                if (tmp == '1') {
                    right[j] = Math.min(right[j], rightSide);
                } else {
                    right[j] = column;
                    rightSide = j;
                }
            }
            for (int j = 0; j < column; j++) {
                result = Math.max(result, (right[j] - left[j]) * height[j]);
            }
        }
        return result;
        // 0 1 1 3 4
        // 2 2 3 4 5
        // --------//
        // 0 1 1 3 0
        // 4 2 4 4 5
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
        for (int j = 0; j < column; j++) {
            dp[0][j] = matrix[0][j] == '1' ? 1 : 0;
            int tmp = matrix[0][j] == '1' ? 1 : 0;
            result = Math.max(result, tmp);
        }
        for (int i = 1; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (matrix[i][j] == '1') {
                    if (j == 0) {
                        dp[i][j] = 1;
                    } else {
                        dp[i][j] = Math.min(dp[i - 1][j - 1], Math.min(dp[i][j - 1], dp[i - 1][j])) + 1;
                    }
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
        if (s1 == null || s2 == null) {
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
                dp[i][j] = (s1.charAt(i - 1) == s3.charAt(i + j - 1) && dp[i - 1][j]) || (s2.charAt(j - 1) == s3.charAt(i + j - 1) && dp[i][j - 1]);
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
        if (t.isEmpty()) {
            return s.length();
        }
        int m = s.length();
        int n = t.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 0; i <= m; i++) {
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
            for (int j = i - 1; j >= 1; j--) {
                List<Integer> prev = result.get(i - 1);

                Integer value = prev.get(j) + prev.get(j - 1);

                tmp.add(value);
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
                int value = result.get(j) + result.get(j - 1);

                result.set(j, value);
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
            return 0;
        }
        int size = triangle.size();
        for (int i = size - 2; i >= 0; i--) {
            List<Integer> current = triangle.get(i);

            List<Integer> prev = triangle.get(i + 1);

            int currentSize = current.size();

            for (int j = 0; j < currentSize; j++) {
                int val = Math.min(prev.get(j), prev.get(j + 1)) + current.get(j);

                current.set(j, val);
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
        for (int num : dp) {
            result += num;
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
        int column = prices.length;
        int[] leftProfit = new int[column];
        int cost = prices[0];
        int leftResult = 0;
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > cost) {
                leftResult = Math.max(leftResult, prices[i] - cost);
            } else {
                cost = prices[i];
            }
            leftProfit[i] = leftResult;
        }
        int[] rightProfit = new int[column + 1];

        int rightResult = 0;
        cost = prices[column - 1];
        for (int i = prices.length - 2; i >= 0; i--) {
            if (prices[i] < cost) {
                rightResult = Math.max(rightResult, cost - prices[i]);
            } else {
                cost = prices[i];
            }
            rightProfit[i] = rightResult;
        }
        int result = 0;
        for (int i = 1; i < column; i++) {
            result = Math.max(result, leftProfit[i] + rightProfit[i + 1]);
        }
        return result;
    }


    /**
     * https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/
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


    /**
     * todo
     * 132. Palindrome Partitioning II
     *
     * @param s
     * @return
     */
    public int minCut(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        }
        int len = s.length();
        int[] dp = new int[len];
        for (int i = 0; i < len; i++) {
            int min = i;
            for (int j = 0; j <= i; j++) {
                if (checkPalindrome(s, j, i)) {
                    min = Math.min(min, j == 0 ? 0 : dp[j - 1] + 1);
                }
            }
            dp[i] = min;
        }
        return dp[len - 1];
    }

    private boolean checkPalindrome(String s, int j, int i) {
        while (j < i) {
            if (s.charAt(i) != s.charAt(j)) {
                return false;
            }
            i--;
            j++;
        }
        return true;
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
            robCurrent = Math.max(robPrev + num, robCurrent);
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
        if (nums == null) {
            return 0;
        }
        if (nums.length == 1) {
            return nums[0];
        }
        return Math.max(intervalRob(nums, 0, nums.length - 2), intervalRob(nums, 1, nums.length - 1));
    }

    private int intervalRob(int[] nums, int start, int end) {
        int robPrev = 0;
        int robCurrent = 0;
        for (int i = start; i <= end; i++) {
            int tmp = robCurrent;
            robCurrent = Math.max(robCurrent, robPrev + nums[i]);
            robPrev = tmp;
        }
        return Math.max(robCurrent, robPrev);
    }


    // 房子颜色问题 //


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
                    dp[i][j] = Math.max(1, Math.min(dp[i][j + 1], dp[i + 1][j]) - dungeon[i][j]);
                }
            }
        }
        return dp[0][0];
    }

    public int calculateMinimumHPII(int[][] dungeon) {
        if (dungeon == null || dungeon.length == 0) {
            return 0;
        }
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
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            int min = i;
            for (int j = 1; j * j <= i; j++) {
                int previous = i - j * j;
                min = Math.min(1 + dp[previous], min);
            }
            dp[i] = min;
        }
        return dp[n];
    }

    /**
     * https://www.lintcode.com/problem/168/solution/27179
     * todo
     * 312. Burst Balloons
     * explains:
     * https://leetcode.com/problems/burst-balloons/discuss/892552/For-those-who-are-not-able-to-understand-any-solution-WITH-DIAGRAM
     *
     * @param nums: A list of integer
     * @return: An integer, maximum coins
     */
    public int maxCoins(int[] nums) {
        // write your code here
        if (nums == null || nums.length == 0) {
            return 0;
        }
        nums = init(nums);

        int[][] dp = new int[nums.length][nums.length];
        for (int i = nums.length - 1; i >= 0; i--) {
            for (int j = i + 2; j < nums.length; j++) {
                for (int k = i + 1; k < j; k++) {
                    dp[i][j] = Math.max(dp[i][j], dp[i][k] + dp[k][j] + nums[i] * nums[k] * nums[j]);
                }
            }
        }
        return dp[0][nums.length - 1];


    }

    public int[] init(int[] nums) {
        //初始化数组，头部和尾部插入1
        int[] temp = new int[nums.length + 2];
        temp[0] = 1;
        for (int i = 1; i < temp.length - 1; i++) {
            temp[i] = nums[i - 1];
        }
        temp[temp.length - 1] = 1;
        return temp;
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
