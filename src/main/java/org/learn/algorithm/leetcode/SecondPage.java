package org.learn.algorithm.leetcode;


import java.util.HashMap;
import java.util.Map;

/**
 * 第二页
 *
 * @author luk
 * @date 2021/4/11
 */
public class SecondPage {


    /**
     * 125. Valid Palindrome
     *
     * @param s
     * @return
     */
    public boolean isPalindrome(String s) {
        if (s == null) {
            return false;
        }
        s = s.trim();
        if (s.isEmpty()) {
            return true;
        }
        char[] words = s.toCharArray();
        int start = 0;
        int end = words.length - 1;
        while (start < end) {
            while (start < end && !Character.isLetterOrDigit(words[start])) {
                start++;
            }
            while (start < end && !Character.isLetterOrDigit(words[end])) {
                end--;
            }
            if (Character.toLowerCase(words[start]) != Character.toLowerCase(words[end])) {
                return false;
            }
            start++;
            end--;
        }
        return true;
    }


    /**
     * 128. Longest Consecutive Sequence
     *
     * @param nums
     * @return
     */
    public int longestConsecutive(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        Map<Integer, Integer> map = new HashMap<>();
        int result = 0;
        for (int num : nums) {
            if (map.containsKey(num)) {
                continue;
            }
            Integer left = map.getOrDefault(num - 1, 0);

            Integer right = map.getOrDefault(num + 1, 0);

            int val = left + right + 1;
            result = Math.max(result, val);

            map.put(num, val);
            map.put(num - left, val);
            map.put(num + right, val);

        }
        return result;

    }


}
