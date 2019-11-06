package org.dora.algorithm.geeksforgeek;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * 递归其实是动态规划的另种方式
 *
 * @author dora
 * @date 2019-04-26
 */
public class DynamicProgramming {
    /**
     * 10. Regular Expression Matching
     * dp[i][j] = dp[i-1][j-1] if s[i] == t[j] || t[j] == '.'
     * = dp[i][j-2] if s[i] != t[j] && t[j-2] != '.' example :  s = a t = ab*
     * = dp[i-1][j] || dp[i][j-2] || dp[i][j-1]
     * e.g case 1: s = aaaaa t = a*;
     * case 2: s = a t = aa*
     * case 3: s = a t = a *
     *
     * @param s
     * @param p
     * @return
     */
    public boolean isMatch(String s, String p) {
        if (s == null && p == null) {
            return false;
        } else if (s == null) {
            return true;
        }
        int m = s.length();
        int n = p.length();

        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[0][0] = true;
        for (int j = 1; j <= n; j++) {
            dp[0][j] = p.charAt(j - 1) == '*' && dp[0][j - 2];
        }

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (s.charAt(i - 1) == p.charAt(j - 1) || p.charAt(j - 1) == '.') {
                    dp[i][j] = dp[i - 1][j - 1];
                } else if (p.charAt(j - 1) == '*') {
                    if (s.charAt(i - 1) != p.charAt(j - 2) && p.charAt(j - 2) != '.') {
                        dp[i][j] = dp[i][j - 2];
                    } else {
                        dp[i][j] = dp[i][j - 1] || dp[i - 1][j] || dp[i][j - 2];
                    }
                }
            }
        }
        return dp[m][n];
    }


    /**
     * 62. Unique Paths
     *
     * @param m
     * @param n
     * @return
     */
    public int uniquePaths(int m, int n) {
        if (m <= 0 || n <= 0) {
            return 0;
        }
        int[] dp = new int[n];
        dp[0] = 1;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                dp[j] = dp[j] + (j > 0 ? dp[j - 1] : 0);
            }
        }
        return dp[n - 1];
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

        int[][] dp = new int[row][column];

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (obstacleGrid[i][j] == 1) {
                    continue;
                }
                if (i == 0 && j == 0) {
                    dp[i][j] = 1;
                } else if (i == 0) {
                    dp[i][j] = dp[i][j - 1];
                } else if (j == 0) {
                    dp[i][j] = dp[i - 1][j];
                } else {
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
                }
            }

        }
        return dp[row - 1][column - 1];
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

        for (int i = 0; i < row; i++) {

            for (int j = 0; j < column; j++) {

                if (i == 0 && j == 0) {
                    dp[j] = grid[0][0];
                } else if (i == 0) {
                    dp[j] = dp[j - 1] + grid[0][j];
                } else if (j == 0) {
                    dp[j] = dp[j] + grid[i][0];
                } else {
                    dp[j] = Math.min(dp[j], dp[j - 1]) + grid[i][j];
                }

            }
        }
        return dp[column - 1];
    }

    /**
     * todo
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

        dp[0][0] = 0;

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
                    dp[i][j] = 1 + Math.min(Math.min(dp[i][j - 1], dp[i - 1][j - 1]), dp[i - 1][j]);
                }
            }
        }
        return dp[m][n];
    }

    /**
     * height[i]: 第i列时的最大高度
     * left[i]: 第i列 的左边界
     * right[i]: 第i列 的右边界
     * 随着层次的遍历 矩形的面积也随着上一层而进行变动
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

        for (int i = 0; i < column; i++) {
            right[i] = column;
        }
        int result = 0;

        for (int i = 0; i < row; i++) {

            int leftSide = 0;

            int rightSide = column;

            for (int j = 0; j < column; j++) {
                if (matrix[i][j] == '1') {
                    height[j]++;
                } else {
                    height[j] = 0;
                }
            }
            for (int j = 0; j < column; j++) {
                if (matrix[i][j] == '1') {
                    left[j] = Math.max(leftSide, left[j]);
                } else {
                    left[j] = 0;
                    leftSide = j + 1;
                }
            }
            for (int j = column - 1; j >= 0; j--) {
                if (matrix[i][j] == '1') {

                    right[j] = Math.min(rightSide, right[j]);
                } else {
                    right[j] = column;

                    rightSide = j;
                }
            }
            for (int j = 0; j < column; j++) {
                result = Math.max(result, height[j] * (right[j] - left[j]));
            }
        }
        return result;
    }

    /**
     * todo 未来考虑使用动态规划
     * 87. Scramble String
     *
     * @param s1
     * @param s2
     * @return
     */
    public boolean isScramble(String s1, String s2) {
        if (s1 == null || s2 == null) {
            return false;
        }

        if (s1.equals(s2)) {
            return true;
        }
        int m = s1.length();
        int n = s2.length();

        if (m != n) {
            return false;
        }
        int[] hash = new int[256];
        for (int i = 0; i < m; i++) {
            hash[s1.charAt(i) - '0']--;
            hash[s2.charAt(i) - '0']++;
        }
        for (int i = 0; i < hash.length; i++) {
            if (hash[i] != 0) {
                return false;
            }
        }
        for (int i = 1; i < m; i++) {
            if (this.isScramble(s1.substring(0, i), s2.substring(0, i)) &&
                    this.isScramble(s1.substring(i), s2.substring(i))) {
                return true;
            }
            if (this.isScramble(s1.substring(i), s2.substring(0, m - i)) &&
                    this.isScramble(s1.substring(0, i), s2.substring(m - i))) {
                return true;
            }
        }
        return false;
    }


    // ----------递归--------------//

    public List<List<Integer>> subsetsWithDup(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new ArrayList<>();
        }
        Arrays.sort(nums);
        List<List<Integer>> ans = new ArrayList<>();
        this.subsetsWithDup(ans, new ArrayList<>(), 0, nums);
        return ans;
    }

    private <E> void subsetsWithDup(List<List<Integer>> ans, List<Integer> tmp, int start, int[] nums) {
        ans.add(new ArrayList<>(tmp));
        for (int i = start; i < nums.length; i++) {
            if (i > start && nums[i] == nums[i - 1]) {
                continue;
            }

            tmp.add(nums[i]);

            this.subsetsWithDup(ans, tmp, i + 1, nums);

            tmp.remove(tmp.size() - 1);
        }
    }

}
