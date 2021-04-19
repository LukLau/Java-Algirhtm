package org.learn.algorithm.leetcode;

import java.util.*;

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


    /**
     * 239. Sliding Window Maximum
     *
     * @param nums
     * @param k
     * @return
     */
    public int[] maxSlidingWindow(int[] nums, int k) {
        if (nums == null || nums.length == 0) {
            return new int[]{};
        }
        LinkedList<Integer> deque = new LinkedList<>();
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < nums.length; i++) {
            int index = i - k + 1;
            if (!deque.isEmpty() && index > deque.peekFirst()) {
                deque.pollFirst();
            }
            while (!deque.isEmpty() && nums[i] >= nums[deque.peekLast()]) {
                deque.pollLast();
            }
            deque.offer(i);
            if (index >= 0) {
                result.add(nums[deque.peekFirst()]);
            }
        }
        int[] ans = new int[result.size()];
        for (int i = 0; i < result.size(); i++) {
            ans[i] = result.get(i);
        }
        return ans;
    }


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
        Map<Character, Integer> map = new HashMap<>();
        int result = 0;
        int left = 0;
        char[] words = s.toCharArray();
        int end = 0;
        while (end < words.length) {
            char word = words[end++];
            Integer num = map.getOrDefault(word, 0);

            map.put(word, num + 1);

            while (map.size() > 2) {
                char leftWord = words[left++];
                Integer tmp = map.get(leftWord);
                tmp--;
                if (tmp == 0) {
                    map.remove(leftWord);
                } else {
                    map.put(leftWord, tmp);
                }
            }
            result = Math.max(result, end - left);
        }
        return result;
        // Write your code here
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
        intervalPartition(result, new ArrayList<>(), 0, s);
        return result;
    }

    private void intervalPartition(List<List<String>> result, List<String> tmp, int start, String s) {
        if (start == s.length()) {
            result.add(new ArrayList<>(tmp));
            return;
        }
        for (int i = start; i < s.length(); i++) {
            if (validPalindrome(s, start, i)) {
                String substring = s.substring(start, i + 1);
                tmp.add(substring);
                intervalPartition(result, tmp, i + 1, s);
                tmp.remove(tmp.size() - 1);
            }
        }
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
        int m = s.length();
        int[] cut = new int[m];
        boolean[][] dp = new boolean[m][m];
        for (int i = 1; i < m; i++) {
            int minCut = i;
            for (int j = 0; j <= i; j++) {
                dp[j][i] = s.charAt(j) == s.charAt(i) && (i - j < 2 || dp[j + 1][i - 1]);
                if (dp[j][i]) {
                    if (j == 0) {
                        minCut = 0;
                    } else {
                        minCut = Math.min(minCut, 1 + cut[j - 1]);
                    }
                }
            }
            cut[i] = minCut;
        }
        return cut[m - 1];
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


    /**
     * 266
     * Palindrome Permutation
     *
     * @param s: the given string
     * @return: if a permutation of the string could form a palindrome
     */
    public boolean canPermutePalindrome(String s) {
        if (s == null || s.isEmpty()) {
            return true;
        }
        boolean odd = false;
        char[] words = s.toCharArray();
        int[] hash = new int[256];
        for (char word : words) {
            hash[word - 'a']++;
        }
        for (int num : hash) {
            if (num % 2 != 0) {
                if (odd) {
                    return false;
                }
                odd = true;
            }
        }
        return true;
    }


    /**
     * @param s: the given string
     * @return: all the palindromic permutations (without duplicates) of it
     */
    public List<String> generatePalindromes(String s) {
        // write your code here
        if (s == null || s.isEmpty()) {
            return new ArrayList<>();
        }
        List<String> result = new ArrayList<>();
        Map<Character, Integer> map = new HashMap<>();
        char[] words = s.toCharArray();
        List<Character> odds = new ArrayList<>();
        for (char word : words) {
            Integer count = map.getOrDefault(word, 0);
            count++;
            map.put(word, count);
        }
        Character oddCharacter = null;
        StringBuilder builder = new StringBuilder();
        Set<Map.Entry<Character, Integer>> entry = map.entrySet();
        for (Map.Entry<Character, Integer> item : entry) {
            Character key = item.getKey();
            Integer count = item.getValue();
            if (count % 2 == 1) {
                if (oddCharacter != null) {
                    return result;
                }
                oddCharacter = key;
            }
            for (int i = 0; i < count / 2; i++) {
                builder.append(key);
            }
        }
        return result;

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


    public static void main(String[] args) {
        StringSolution solution = new StringSolution();
        solution.canPermutePalindrome("aab");
    }


}
