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
        int[] nums = new int[]{2, 1, 4};
        solution.maxProfitIII(nums);
    }

    // 普通动态规划问题

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
        int len = s.length();
        int[] dp = new int[len + 1];
        dp[0] = 1;
        dp[1] = s.charAt(0) == '0' ? 0 : 1;
        for (int i = 2; i <= len; i++) {
            int first = Integer.parseInt(s.substring(i - 1, i));
            int second = Integer.parseInt(s.substring(i - 2, i));
            if (first >= 1 && first <= 9) {
                dp[i] += dp[i - 1];
            }
            if (second >= 1 && second <= 26) {
                dp[i] += dp[i - 2];
            }
        }
        return dp[len];
    }


    // 序列问题

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
        return intervalTotalQueens(dp, 0, n);
    }

    private int intervalTotalQueens(int[] dp, int row, int n) {
        int count = 0;
        if (row == n) {
            count++;
            return count;
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

    // --编辑距离问题 //

    private boolean isValidTotalQueens(int[] dp, int col, int row, int n) {
        for (int i = row - 1; i >= 0; i--) {
            if (dp[i] == col || Math.abs(dp[i] - col) == Math.abs(i - row)) {
                return false;
            }
        }
        return true;
    }


    // ----- //

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
                    dp[i][j] = Math.min(Math.min(dp[i - 1][j - 1], dp[i - 1][j]), dp[i][j - 1]) + 1;
                }
            }
        }
        return dp[m][n];
    }

    /**
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
        int[] height = new int[column];
        int[] left = new int[column];
        int[] right = new int[column];
        Arrays.fill(right, column);
        int result = 0;
        for (int i = 0; i < row; i++) {
            int leftEdge = 0;
            int rightEdge = column;
            for (int j = 0; j < column; j++) {
                char tmp = matrix[i][j];
                if (tmp == '1') {
                    height[j]++;
                    left[j] = Math.max(left[j], leftEdge);
                } else {
                    height[j] = 0;
                    left[j] = leftEdge;
                    leftEdge = j + 1;
                }
            }
            for (int j = column - 1; j >= 0; j--) {
                char tmp = matrix[i][j];
                if (tmp == '1') {
                    right[j] = Math.min(right[j], rightEdge);
                } else {
                    right[j] = column;
                    rightEdge = j;
                }
            }
            for (int j = 0; j < column; j++) {
                if (height[j] != 0) {
                    result = Math.max(result, (right[j] - left[j]) * height[j]);
                }
            }
        }
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
        if (s == null || t == null) {
            return 0;
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
                int val = result.get(i - 1).get(j) + result.get(i - 1).get(j - 1);
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
        List<Integer> result = new ArrayList<>(rowIndex);
        for (int i = 0; i <= rowIndex; i++) {
            for (int j = i - 1; j >= 1; j--) {
                int val = result.get(j) + result.get(j - 1);
                result.set(j, val);
            }
            result.add(1);
        }
        return result;
    }

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
        List<Integer> row = triangle.get(size - 1);
        for (int i = size - 2; i >= 0; i--) {
            List<Integer> current = triangle.get(i);
            int len = current.size();
            for (int j = 0; j < len; j++) {
                int val = Math.min(row.get(j), row.get(j + 1)) + current.get(j);
                row.set(j, val);
            }
        }
        return row.get(0);
    }


    // -卖股票系列问题- //

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
        int cost = prices[0];
        int result = 0;
        for (int i = 1; i < prices.length; i++) {
            int price = prices[i];
            if (price > cost) {
                result = Math.max(result, price - cost);
            } else {
                cost = price;
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
        int[] left = new int[column];
        int[] right = new int[column + 1];
        int cost = prices[0];
        int leftProfit = 0;
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] < cost) {
                cost = prices[i];
            }
            leftProfit = Math.max(leftProfit, prices[i] - cost);
            left[i] = leftProfit;
        }
        cost = prices[column - 1];
        int rightProfit = 0;
        for (int i = column - 2; i >= 0; i--) {
            if (prices[i] > cost) {
                cost = prices[i];
            }
            rightProfit = Math.max(rightProfit, cost - prices[i]);
            right[i] = rightProfit;
        }
        int result = 0;
        for (int i = 1; i < column; i++) {
            result = Math.max(result, left[i] + right[i + 1]);
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
            int cost = -prices[0];
            for (int j = 1; j < prices.length; j++) {
                dp[i][j] = Math.max(dp[i][j - 1], cost + prices[j]);
                cost = Math.max(cost, dp[i - 1][j - 1] - prices[j]);
            }
        }
        return dp[k][prices.length - 1];
    }


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


    // 房屋抢劫系列


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
        if (nums == null || nums.length == 0) {
            return 0;
        }
        if (nums.length == 1) {
            return 1;
        }
        return Math.max(intervalRob(nums, 1, nums.length - 1), intervalRob(nums, 0, nums.length - 2));
    }

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

    // 房子颜色问题

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
            costs[i][0] = Math.min(costs[i - 1][1], costs[i - 1][2]) + costs[i][0];
            costs[i][1] = Math.min(costs[i - 1][2], costs[i - 1][0]) + costs[i][1];
            costs[i][2] = Math.min(costs[i - 1][1], costs[i - 1][0]) + costs[i][2];
        }
        return Math.min(Math.min(costs[row - 1][0], costs[row - 1][1]), costs[row - 1][2]);
    }


    /**
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
        int firstIndex = -1;
        int secondIndex = -1;
        int row = costs.length;
        int column = costs[0].length;
        for (int i = 0; i < row; i++) {
            int index1 = -1;
            int index2 = -1;
            for (int j = 0; j < column; j++) {
                if (firstIndex >= 0) {
                    if (j != firstIndex) {
                        costs[i][j] += costs[i - 1][firstIndex];
                    } else {
                        costs[i][j] += costs[i - 1][secondIndex];
                    }
                }
                if (index1 == -1 || costs[i][j] < costs[i][index1]) {
                    index2 = index1;
                    index1 = j;
                } else if (index2 == -1 || costs[i][j] < costs[i][index2]) {
                    index2 = j;
                }
            }
            firstIndex = index1;
            secondIndex = index2;
        }
        return costs[row - 1][firstIndex];
    }


    // 普通题

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
        if (n == 1) {
            return 1;
        }
        int[] dp = new int[n + 1];
        dp[0] = 0;
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            int tmp = i;
            for (int j = 1; j * j <= i; j++) {
                tmp = Math.min(tmp, 1 + dp[i - j * j]);
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
}
