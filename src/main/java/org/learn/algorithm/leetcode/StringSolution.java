package org.learn.algorithm.leetcode;

import java.util.HashMap;
import java.util.Map;

/**
 * 字符串系列问题
 *
 * @author luk
 * @date 2021/4/6
 */
public class StringSolution {


    // 滑动窗口系列//

    /**
     * 76. Minimum Window Substring
     *
     * @param s
     * @param t
     * @return
     */
    public String minWindow(String s, String t) {
        if (s == null || t == null) {
            return "";
        }
        int[] hash = new int[256];
        int n = t.length();

        for (int i = 0; i < n; i++) {
            hash[t.charAt(i)]++;
        }
        int begin = 0;
        int head = 0;
        int result = Integer.MAX_VALUE;
        int end = 0;
        int m = s.length();
        while (end < m) {
            if (hash[s.charAt(end++)]-- > 0) {
                n--;
            }
            while (n == 0) {
                if (end - begin < result) {
                    result = end - begin;
                    head = begin;
                }
                if (hash[s.charAt(begin++)]++ == 0) {
                    n++;
                }
            }
        }
        if (result != Integer.MAX_VALUE) {
            return s.substring(head, head + result);
        }
        return "";
    }


    // 正则表达式//

    /**
     * 10. Regular Expression Matching
     *
     * @param s
     * @param p
     * @return
     */
    public boolean isMatch(String s, String p) {
        if (p.isEmpty()) {
            return s.isEmpty();
        }
        int m = s.length();

        int n = p.length();

        boolean[][] dp = new boolean[m + 1][n + 1];

        dp[0][0] = true;

        for (int j = 1; j <= n; j++) {
            dp[0][j] = p.charAt(j - 1) == '*' && dp[0][j - 2];
        }
        char[] words = s.toCharArray();
        char[] tmp = p.toCharArray();

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (words[i - 1] == tmp[j - 1] || tmp[j - 1] == '.') {
                    dp[i][j] = dp[i - 1][j - 1];
                } else if (tmp[j - 1] == '*') {
                    if (words[i - 1] != tmp[j - 2] && tmp[j - 2] != '.') {
                        dp[i][j] = dp[i][j - 2];
                    } else {
                        dp[i][j] = dp[i - 1][j] || dp[i][j - 1] || dp[i][j - 2];
                    }
                }
            }
        }
        return dp[m][n];
    }

    /**
     * todo
     * 44. Wildcard Matching
     *
     * @param s
     * @param p
     * @return
     */
    public boolean isMatchII(String s, String p) {
        if (p.isEmpty()) {
            return s.isEmpty();
        }
        int m = s.length();
        int n = p.length();
        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[0][0] = true;
        for (int j = 1; j <= n; j++) {
            dp[0][j] = p.charAt(j - 1) == '*' && dp[0][j - 1];
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (s.charAt(i - 1) == p.charAt(j - 1) || p.charAt(j - 1) == '?') {
                    dp[i][j] = dp[i - 1][j - 1];
                } else if (p.charAt(j - 1) == '*') {
                    dp[i][j] = dp[i - 1][j] || dp[i][j - 1];
                }
            }
        }
        return dp[m][n];

    }

    // 重复字符串问题 //

    /**
     * 3. Longest Substring Without Repeating Characters
     *
     * @param s
     * @return
     */
    public int lengthOfLongestSubstring(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        }
        int result = 0;
        Map<Character, Integer> map = new HashMap<>();
        int left = 0;
        char[] words = s.toCharArray();
        for (int i = 0; i < words.length; i++) {
            if (map.containsKey(words[i])) {
                left = Math.max(left, map.get(words[i]) + 1);
            }

            map.put(words[i], i);
            result = Math.max(result, i - left + 1);
        }
        return result;
    }


    public int lengthOfLongestSubstringII(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        }
        int[] hash = new int[256];
        int left = 0;
        char[] words = s.toCharArray();
        int result = 0;
        for (int i = 0; i < words.length; i++) {
            left = Math.max(left, hash[s.charAt(i)]);

            result = Math.max(result, i - left + 1);

            hash[s.charAt(i)] = i + 1;

        }
        return result;
    }

    //--回文系列//

    public String longestPalindrome(String s) {
        if (s == null || s.isEmpty()) {
            return "";
        }

        int len = s.length();
        boolean[][] dp = new boolean[len][len];
        int result = Integer.MIN_VALUE;
        int left = 0;
        for (int i = 0; i < len; i++) {
            for (int j = 0; j <= i; j++) {
                if (s.charAt(j) == s.charAt(i) && (i - j < 2 || dp[j + 1][i - 1])) {
                    dp[j][i] = true;
                }
                if (dp[j][i] && i - j + 1 > result) {
                    left = j;
                    result = i - j + 1;
                }
            }
        }
        if (result != Integer.MIN_VALUE) {
            return s.substring(left, left + result);
        }
        return "";
    }


    private int longestPalindrome = Integer.MIN_VALUE;
    private int palindrome = 0;

    public String longestPalindromeII(String s) {
        if (s == null || s.isEmpty()) {
            return "";
        }
        int len = s.length();
        char[] words = s.toCharArray();

        for (int i = 0; i < len; i++) {
            intervalPalindrome(words, i, i);
            intervalPalindrome(words, i, i + 1);
        }
        if (longestPalindrome != Integer.MIN_VALUE) {
            return s.substring(longestPalindrome, palindrome + longestPalindrome);
        }
        return "";
    }

    private void intervalPalindrome(char[] words, int start, int end) {
        while (start >= 0 && end < words.length && words[start] == words[end]) {
            if (end - start + 1 > longestPalindrome) {
                longestPalindrome = end - start + 1;
                palindrome = start;
            }
            start--;
            end++;
        }
    }


    /**
     * 9. Palindrome Number
     *
     * @param x
     * @return
     */
    public boolean isPalindrome(int x) {
        if (x <= 0) {
            return x == 0;
        }
        String word = String.valueOf(x);

        int start = 0;

        int end = word.length() - 1;

        while (start < end) {
            if (word.charAt(start) != word.charAt(end)) {
                return false;
            }
            start++;
            end--;
        }
        return true;
    }


    public static void main(String[] args) {
        StringSolution solution = new StringSolution();
        solution.isPalindrome(11);
    }


}
