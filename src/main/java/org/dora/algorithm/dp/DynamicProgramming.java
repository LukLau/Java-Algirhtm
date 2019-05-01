package org.dora.algorithm.dp;

/**
 * @author dora
 * @date 2019-04-26
 */
public class DynamicProgramming {
    /**
     * 10. Regular Expression Matching
     * dp[i][j] = dp[i-1][j-1] if s[i] == t[j] || t[j] == '.'
     *          = dp[i][j-2] if s[i] != t[j] && t[j-2] != '.' example :  s = a t = ab*
     *          = dp[i-1][j] || dp[i][j-2] || dp[i][j-1]
     *          e.g case 1: s = aaaaa t = a*;
     *              case 2: s = a t = aa*
     *              case 3: s = a t = a *
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
}
