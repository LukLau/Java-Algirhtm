package org.dora.algorithm.geeksforgeek;

/**
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

}
