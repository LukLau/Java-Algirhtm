package org.learn.algorithm.leetcode;

import org.learn.algorithm.datastructure.TreeNode;

import java.util.*;

/**
 * Vip 题目
 *
 * @author luk
 * @date 2021/4/12
 */
public class VipSolution {

    /**
     * 156 Binary Tree Upside Down
     *
     * @param root: the root of binary tree
     * @return: new root
     */
    public TreeNode upsideDownBinaryTree(TreeNode root) {
        if (root == null || root.left == null) {
            return root;
        }
        TreeNode node = upsideDownBinaryTree(root.left);

        root.left.left = root.right;

        root.left.right = root;

        root.left = null;

        root.right = null;

        return node;
        // write your code here
    }


    /**
     * 161 One Edit Distance
     *
     * @param s: a string
     * @param t: a string
     * @return: true if they are both one edit distance apart or false
     */
    public boolean isOneEditDistance(String s, String t) {
        if (s == null || t == null) {
            return false;
        }
        if (s.equals(t)) {
            return false;
        }
        int m = s.length();
        int n = t.length();
        int len = Math.min(m, n);
        for (int i = 0; i < len; i++) {
            if (s.charAt(i) != t.charAt(i)) {
                if (m == n) {
                    return s.substring(i + 1).equals(t.substring(i + 1));
                } else if (m < n) {
                    return s.substring(i).equals(t.substring(i + 1));
                } else {
                    return s.substring(i + 1).equals(t.substring(i));
                }
            }
        }
        return Math.abs(m - n) <= 1;

        // write your code here
    }


    /**
     * todo
     * #163 Missing Ranges
     *
     * @param nums:  a sorted integer array
     * @param lower: An integer
     * @param upper: An integer
     * @return: a list of its missing ranges
     */
    public List<String> findMissingRanges(int[] nums, int lower, int upper) {
        // write your code here
        if (nums == null) {
            return new ArrayList<>();
        }
        List<String> result = new ArrayList<>();
        for (int num : nums) {
            if (num > lower && num >= lower + 1) {
                String tmp = range(lower, num - 1);
                result.add(tmp);
            }
            if (num == upper) {
                return result;
            }
            lower = num + 1;
        }
        if (lower <= upper) {
            result.add(range(lower, upper));
        }
        return result;
    }

    private String range(int lower, int upper) {
        return lower == upper ? String.valueOf(lower) : lower + "->" + upper;
    }


    /**
     * 186 Reverse Words in a String II
     * Medium
     *
     * @param str: a string
     * @return: return a string
     */
    public char[] reverseWords(char[] str) {
        // write your code here
        if (str == null || str.length == 0) {
            return new char[]{};
        }
        int endIndex = 0;
        while (endIndex < str.length) {
            int startIndex = endIndex;
            while (endIndex < str.length && str[endIndex] != ' ') {
                endIndex++;
            }
            if (endIndex == str.length || str[endIndex] == ' ') {
                reverseArray(str, startIndex, endIndex);
            }
            endIndex++;
        }
        reverseArray(str, 0, str.length);
        return str;
    }

    private void reverseArray(char[] str, int start, int end) {
        for (int i = start; i <= (start + end - 1) / 2; i++) {
            swap(str, i, start + end - 1 - i);
        }
    }

    private void swap(char[] str, int i, int j) {
        char tmp = str[i];
        str[i] = str[j];
        str[j] = tmp;
    }

    // 单词最短距离


    /**
     * #243 Shortest Word Distance
     *
     * @param words
     * @param word1
     * @param word2
     * @return
     */
    public int shortestDistance(String[] words, String word1, String word2) {
        // Write your code here
        int index1 = -1;
        int index2 = -1;
        int result = Integer.MAX_VALUE;
        for (int i = 0; i < words.length; i++) {
            String word = words[i];
            if (word.equals(word1)) {
                index1 = i;
            } else if (word.equals(word2)) {
                index2 = i;
            }
            if (index1 != -1 && index2 != -1) {
                result = Math.min(result, Math.abs(index1 - index2));
            }
        }
        return result;
    }

    // 反转数系列


    /**
     * 246 Strobogrammatic Number
     *
     * @param num
     * @return
     */
    public boolean isStrobogrammatic(String num) {
        if (num == null || num.isEmpty()) {
            return false;
        }
        Map<Character, Character> map = getNum();
        int len = num.length();
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < len; i++) {
            char tmp = num.charAt(i);
            Character reverse = map.get(tmp);
            if (reverse == null) {
                return false;
            }
            builder.append(reverse);
        }
        return num.equals(builder.reverse().toString());
    }

    private Map<Character, Character> getNum() {
        Map<Character, Character> map = new HashMap<>();
        map.put('1', '1');
        map.put('6', '9');
        map.put('8', '8');
        map.put('9', '6');
        map.put('0', '0');
        return map;
    }


    /**
     * #247 Strobogrammatic Number II
     * Medium
     *
     * @param n: the length of strobogrammatic number
     * @return: All strobogrammatic numbers
     */
    public List<String> findStrobogrammatic(int n) {
        // write your code here
        if (n < 0) {
            return new ArrayList<>();
        }
        if (n == 1) {
            return Arrays.asList("0", "1", "8");
        }
        List<String> result = new ArrayList<>();
        intervalFind(result, "", n);
        intervalFind(result, "1", n);
        intervalFind(result, "8", n);
        intervalFind(result, "0", n);
        return result;
    }

    private void intervalFind(List<String> result, String s, int n) {
        if (s.length() == n) {
            result.add(s);
            return;
        }
        if (s.length() > n - 2) {
            return;
        }

        if (s.length() != n - 2) {
            intervalFind(result, "0" + s + "0", n);
        }
        intervalFind(result, "1" + s + "1", n);
        intervalFind(result, "6" + s + "9", n);
        intervalFind(result, "8" + s + "8", n);
        intervalFind(result, "9" + s + "6", n);
    }


    public static void main(String[] args) {
        VipSolution solution = new VipSolution();
        String s = "the sky is blue";

        solution.findStrobogrammatic(3);

    }


}
