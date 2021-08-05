package org.learn.algorithm.leetcode;

import java.util.*;

/**
 * 字符串系列问题
 *
 * @author luk
 * @date 2021/4/6
 */
public class StringSolution {

    // 子序列问题


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
        int[] hash = new int[512];
        int n = t.length();
        for (int i = 0; i < n; i++) {
            hash[t.charAt(i)]++;
        }
        int result = Integer.MAX_VALUE;
        int beginIndex = 0;
        int head = 0;
        int endIndex = 0;
        int m = s.length();
        while (endIndex < m) {
            if (hash[s.charAt(endIndex++)]-- > 0) {
                n--;
            }
            while (n == 0) {
                if (endIndex - beginIndex < result) {
                    head = beginIndex;
                    result = endIndex - beginIndex;
                }
                if (hash[s.charAt(beginIndex++)]++ == 0) {
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
        LinkedList<Integer> linkedList = new LinkedList<Integer>();
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < nums.length; i++) {
            int index = i - k + 1;
            if (!linkedList.isEmpty() && linkedList.peekFirst() < index) {
                linkedList.poll();
            }
            while (!linkedList.isEmpty() && nums[linkedList.peekLast()] <= nums[i]) {
                linkedList.pollLast();
            }
            linkedList.offer(i);
            if (index >= 0) {
                result.add(nums[linkedList.peekFirst()]);
            }
        }
        int[] tmp = new int[result.size()];
        for (int i = 0; i < result.size(); i++) {
            tmp[i] = result.get(i);
        }
        return tmp;
    }


    /**
     * 159.Longest Substring with At Most Two Distinct Characters
     *
     * @param s: a string
     * @return: the length of the longest substring T that contains at most 2 distinct characters
     */
    public int lengthOfLongestSubstringTwoDistinct(String s) {
        // Write your code here
        if (s == null || s.isEmpty()) {
            return 0;
        }
        Map<Character, Integer> map = new HashMap<>();
        int result = 0;
        int left = 0;
        char[] words = s.toCharArray();
        for (int i = 0; i < words.length; i++) {
            char tmp = words[i];
            Integer count = map.getOrDefault(tmp, 0);
            map.put(tmp, count + 1);
            while (map.size() > 2) {
                char leftEdge = words[left++];
                Integer leftCount = map.get(leftEdge);
                leftCount--;
                if (leftCount == 0) {
                    map.remove(leftEdge);
                } else {
                    map.put(leftEdge, leftCount);
                }
            }
            result = Math.max(result, i - left + 1);
        }
        return result;
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
            dp[0][j] = p.charAt(j - 1) == '*';
        }
        return false;
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
        int left = 0;

        Map<Character, Integer> map = new HashMap<>();
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


    public int lengthOfLongestSubstringII(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        }
        int result = 0;
        int[] hash = new int[512];
        int left = 0;
        char[] words = s.toCharArray();
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
        int m = s.length();
        boolean[][] dp = new boolean[m][m];
        int result = 0;
        int head = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j <= i; j++) {
                if (s.charAt(j) == s.charAt(i) && (i - j <= 2 || dp[j + 1][i + 1])) {
                    dp[j][i] = true;
                }
                if (dp[j][i] && i - j + 1 > result) {
                    result = i - j + 1;

                    head = j;
                }
            }
        }
        if (result != 0) {
            return s.substring(head, head + result);
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
        int m = s.length();
        for (int i = start; i < m; i++) {
            if (validPalindrome(s, start, i)) {
                tmp.add(s.substring(start, i + 1));
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
        dp[0][0] = true;
        for (int i = 1; i < m; i++) {
            int min = i;
            for (int j = 0; j <= i; j++) {
                dp[j][i] = s.charAt(j) == s.charAt(i) && (i - j < 2 || dp[j + 1][i - 1]);
                if (dp[j][i]) {
                    int val = j == 0 ? 0 : 1 + cut[j - 1];
                    min = Math.min(min, val);
                }
            }
            cut[i] = min;
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
        List<String> result = new ArrayList<>();
        Map<Character, Integer> map = new HashMap<>();
        char[] words = s.toCharArray();
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
        char[] items = builder.toString().toCharArray();
        List<String> itemList = new ArrayList<>();
        constructItem(itemList, 0, items);
        for (String item : itemList) {
            String reverse = new StringBuilder(item).reverse().toString();
            result.add(item + (oddCharacter == null ? reverse : oddCharacter + reverse));
        }
        return result;
    }

    private void constructItem(List<String> itemList, int start, char[] items) {
        if (start == items.length) {
            itemList.add(String.valueOf(items));
            return;
        }
        for (int i = start; i < items.length; i++) {
            if (i > start && items[i] == items[i - 1]) {
                continue;
            }
            swapItem(items, i, start);
            constructItem(itemList, start + 1, items);
            swapItem(items, i, start);
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


    public static void main(String[] args) {
        StringSolution solution = new StringSolution();
        solution.partition("aab");
    }


}
