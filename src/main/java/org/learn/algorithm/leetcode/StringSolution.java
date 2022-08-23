package org.learn.algorithm.leetcode;


import org.learn.algorithm.datastructure.ListNode;

import java.util.*;

/**
 * 字符串系列问题
 *
 * @author luk
 * @date 2021/4/6
 */
public class StringSolution {

    // 最长公共字串系列


    private int longestPalindrome = Integer.MIN_VALUE;


    // 子序列问题

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
                if (nums[j] < nums[i] && dp[i] < dp[j] + 1) {
                    dp[i] = dp[j] + 1;
                }

            }
        }
        int result = 0;
        for (int num : dp) {
            result = Math.max(result, num);
        }
        return result;
    }


    // 滑动窗口系列//
    private int palindrome = 0;

    public static void main(String[] args) {
        StringSolution solution = new StringSolution();
//        solution.lengthOfLongestSubstringTwoDistinct("eceba");

        String s = "mississippi";
        String p = "m??*ss*?i*pi";

        solution.isMatchII(s, p);

    }

    /**
     * todo
     * longest common substring
     *
     * @param str1 string字符串 the string
     * @param str2 string字符串 the string
     * @return string字符串
     */
    public String longestCommonSubstring(String str1, String str2) {
        if (str1 == null || str2 == null) {
            return "";
        }
        int m = str1.length();
        int n = str2.length();
        int[][] dp = new int[m + 1][n + 1];
        int result = 0;
        int index = 0;
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (str1.charAt(i - 1) == str2.charAt(j - 1)) {
                    dp[i][j] = 1 + dp[i - 1][j - 1];
                    if (dp[i][j] > result) {
                        index = i;
                        result = dp[i][j];
                    }
                }
            }
        }
        if (result == 0) {
            return "";
        }
        return str1.substring(index - result, index);
    }

    // 正则表达式//

    // 滑动窗口系列 //

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
        int result = Integer.MAX_VALUE;
        int begin = 0;
        int head = 0;
        int endIndex = 0;
        char[] words = s.toCharArray();
        while (endIndex < words.length) {
            if (hash[words[endIndex++]]-- > 0) {
                n--;
            }
            while (n == 0) {
                if (endIndex - begin < result) {
                    head = begin;
                    result = endIndex - begin;
                }
                if (hash[s.charAt(begin++)]++ == 0) {
                    n++;
                }
            }
        }
        if (result == Integer.MAX_VALUE) {
            return "";
        }
        return s.substring(head, head + result);
    }

    /**
     * 239. Sliding Window Maximum
     *
     * @param nums
     * @param k
     * @return
     */
    public int[] maxSlidingWindow(int[] nums, int k) {
        if (nums == null || nums.length == 0) {
            return null;
        }
        LinkedList<Integer> linkedList = new LinkedList<>();
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < nums.length; i++) {
            int index = i - k + 1;
            if (!linkedList.isEmpty() && linkedList.peekFirst() < index) {
                linkedList.pollFirst();
            }
            while (!linkedList.isEmpty() && nums[linkedList.peekLast()] <= nums[i]) {
                linkedList.pollLast();
            }
            linkedList.offer(i);
            if (index >= 0 && !linkedList.isEmpty()) {
                result.add(nums[linkedList.peekFirst()]);
            }
        }
        int[] tmp = new int[result.size()];
        for (int i = 0; i < result.size(); i++) {
            tmp[i] = result.get(i);
        }
        return tmp;
    }

    // 重复字符串问题 //

    /**
     * 159.Longest Substring with At Most Two Distinct Characters
     *
     * @param s: a string
     * @return: the length of the longest substring T that contains at most 2 distinct characters
     */
    public int lengthOfLongestSubstringTwoDistinct(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        }
        int result = 0;
        int left = 0;
        Map<Character, Integer> map = new HashMap<>();
        char[] words = s.toCharArray();
        for (int i = 0; i < words.length; i++) {
            char tmp = s.charAt(i);
            int num = map.getOrDefault(tmp, 0);
            map.put(tmp, num + 1);

            while (map.size() > 2) {
                Integer count = map.get(s.charAt(left));
                count--;
                if (count == 0) {
                    map.remove(s.charAt(left));
                } else {
                    map.put(s.charAt(left), count);
                }
                left++;
            }
            result = Math.max(result, i - left + 1);
        }
        return result;
    }


    //--魔法匹配系列//


    /**
     * 10. Regular Expression Matching
     *
     * @param s
     * @param p
     * @return
     */
    public boolean isMatch(String s, String p) {
        if (s.isEmpty()) {
            return p.isEmpty();
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
                    if (p.charAt(j - 2) != '.' && s.charAt(i - 1) != p.charAt(j - 2)) {
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
                    dp[i][j] = dp[i - 1][j] || dp[i][j - 1] || dp[i][j - 2];
                }
            }
        }
        return dp[m][n];
    }

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
        Map<Character, Integer> map = new HashMap<>();
        int left = 0;
        int result = 0;
        char[] words = s.toCharArray();
        for (int i = 0; i < words.length; i++) {
            if (map.containsKey(words[i])) {
                left = Math.max(left, map.get(words[i]) + 1);
            }
            result = Math.max(result, i - left + 1);
            map.put(words[i], i);

        }
        return result;
    }

    public int lengthOfLongestSubstringii(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        }

        int[] hash = new int[256];
        char[] words = s.toCharArray();
        int result = 0;
        int left = 0;
        for (int i = 0; i < words.length; i++) {
            left = Math.max(left, hash[s.charAt(i)]);

            result = Math.max(result, i - left + 1);

            hash[s.charAt(i)] = i + 1;

        }
        return result;
    }

    public String longestPalindrome(String s) {
        if (s == null || s.isEmpty()) {
            return "";
        }
        int result = Integer.MIN_VALUE;
        int len = s.length();
        int begin = 0;
        boolean[][] dp = new boolean[len][len];

        for (int i = 0; i < len; i++) {
            for (int j = 0; j <= i; j++) {
                if (s.charAt(j) == s.charAt(i) && (i - j <= 2 || dp[j + 1][i - 1])) {
                    dp[j][i] = true;
                }
                if (dp[j][i] && i - j + 1 > result) {
                    result = i - j + 1;
                    begin = j;
                }
            }
        }
        if (result == Integer.MIN_VALUE) {
            return "";
        }
        return s.substring(begin, begin + result);
    }


    /**
     * 205. Isomorphic Strings
     *
     * @param s
     * @param t
     * @return
     */
    public boolean isIsomorphic(String s, String t) {
        if (s == null || t == null) {
            return false;
        }
        int m = s.length();
        int n = t.length();
        Map<Character, Integer> map1 = new HashMap<>();
        Map<Character, Integer> map2 = new HashMap<>();
        for (int i = 0; i < m; i++) {
            int idx1 = map1.getOrDefault(s.charAt(i), i);
            int idx2 = map2.getOrDefault(t.charAt(i), i);
            if (idx1 != idx2) {
                return false;
            }
            map1.put(s.charAt(i), idx1);
            map2.put(t.charAt(i), idx2);
        }
        return true;
    }


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
     * 131. Palindrome Partitioning
     *
     * @param s
     * @return
     */
    public List<List<String>> partition(String s) {
        if (s == null || s.isEmpty()) {
            return new ArrayList<>();
        }
        List<List<String>> result = new ArrayList<>();
        internalPartition(result, new ArrayList<>(), s);
        return result;
    }

    private void internalPartition(List<List<String>> result, List<String> tmp, String s) {
        if (s.isEmpty()) {
            result.add(new ArrayList<>(tmp));
            return;
        }
        for (int i = 0; i < s.length(); i++) {
            if (validPalindrome(s, 0, i)) {
                tmp.add(s.substring(0, i + 1));
                internalPartition(result, tmp, s.substring(i + 1));
                tmp.remove(tmp.size() - 1);
            }
        }
    }

    private boolean validPalindrome(char[] words, int i, int j) {
        if (i > j) {
            return false;
        }
        while (i < j) {
            if (words[i] != words[j]) {
                return false;
            }
            i++;
            j--;
        }
        return true;
    }

    private boolean validPalindrome(String s, int i, int j) {
        if (i > j) {
            return false;
        }
        while (i < j) {
            if (s.charAt(i) != s.charAt(j)) {
                return false;
            }
            i++;
            j--;
        }
        return true;
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
        int[] dp = new int[len + 1];
        for (int i = 0; i < len; i++) {
            int min = i;
            for (int j = 0; j < i; j++) {
                if (validPalindrome(s, j, i)) {
                    min = j == 0 ? 0 : Math.min(min, 1 + dp[j - 1]);
                }
            }
            dp[i] = min;
        }
        return dp[len - 1];
    }

    /**
     * 9. Palindrome Number
     *
     * @param x
     * @return
     */
    public boolean isPalindrome(int x) {
        if (x < 0) {
            return false;
        }
        int result = 0;
        while (x > result) {
            result = result * 10 + x % 10;
            x /= 10;
        }
        return result / 10 == x || result == x;
    }

    /**
     * 266
     * Palindrome Permutation
     *
     * @param s: the given string
     * @return: if a permutation of the string could form a palindrome
     */
    public boolean canPermutePalindrome(String s) {
        if (s == null || s.isEmpty()) {
            return false;
        }
        Map<Character, Integer> map = new HashMap<>();
        char[] words = s.toCharArray();
        for (char word : words) {
            Integer count = map.getOrDefault(word, 0);
            count++;
            map.put(word, count);
        }
        boolean existOdd = false;
        for (Map.Entry<Character, Integer> item : map.entrySet()) {
            Integer count = item.getValue();
            if (count % 2 != 0) {
                if (existOdd) {
                    return false;
                }
                existOdd = true;
            }

        }
        return true;
    }

    /**
     * 267
     * Palindrome Permutation II
     *
     * @param s: the given string
     * @return: all the palindromic permutations (without duplicates) of it
     */
    public List<String> generatePalindromes(String s) {
        // write your code here
        if (s == null || s.isEmpty()) {
            return new ArrayList<>();
        }
        Map<Character, Integer> map = new HashMap<>();
        char[] words = s.toCharArray();
        Character oddEven = null;
        for (char word : words) {
            Integer count = map.getOrDefault(word, 0);
            map.put(word, count + 1);
        }
        StringBuilder builder = new StringBuilder();
        for (Map.Entry<Character, Integer> item : map.entrySet()) {
            Integer count = item.getValue();
            Character key = item.getKey();
            if (count % 2 != 0) {
                if (oddEven != null) {
                    return new ArrayList<>();
                }
                oddEven = key;
            }
            for (int i = 0; i < count / 2; i++) {
                builder.append(key);
            }
        }
        List<String> result = new ArrayList<>();
        boolean[] used = new boolean[builder.toString().length()];

        intervalPalindrome(oddEven, used, builder.toString().toCharArray(), new StringBuilder(), result);

        return result;
    }

    private void intervalPalindrome(Character odd, boolean[] used, char[] words, StringBuilder builder, List<String> ans) {
        if (builder.length() == words.length) {
            String item = builder.toString() + (odd != null ? odd : "") + builder.reverse().toString();
            ans.add(item);
            builder.reverse();
            return;
        }
        for (int i = 0; i < words.length; i++) {
            if (i > 0 && (words[i] == words[i - 1]) && !used[i - 1]) {
                continue;
            }
            if (!used[i]) {
                used[i] = true;
                builder.append(words[i]);
                intervalPalindrome(odd, used, words, builder, ans);
                used[i] = false;
                builder.deleteCharAt(builder.length() - 1);
            }
        }
    }

    private void swapItem(char[] words, int start, int end) {
        char tmp = words[start];
        words[start] = words[end];
        words[end] = tmp;
    }

    /**
     * todo kmp
     * 214. Shortest Palindrome
     *
     * @param s
     * @return
     */
    public String shortestPalindrome(String s) {
        if (s == null || s.isEmpty()) {
            return "";
        }
        String reverse = new StringBuilder(s).reverse().toString();
        return "";
    }


}
