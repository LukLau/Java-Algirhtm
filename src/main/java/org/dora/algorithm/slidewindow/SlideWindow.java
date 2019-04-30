package org.dora.algorithm.slidewindow;

import java.util.HashMap;

/**
 * @author dora
 * @date 2019-05-01
 */
public class SlideWindow {

    public static void main(String[] args) {
        SlideWindow slideWindow = new SlideWindow();

//        int result = slideWindow.lengthOfLongestSubstringTwoDistinct("a");
//        System.out.println(result);
//        System.out.println(slideWindow.lengthOfLongestSubstringTwoDistinct("ab"));
//        System.out.println(slideWindow.lengthOfLongestSubstringTwoDistinct("abbcd"));
        String s = "ABCADEF";
        String t = "ABC";
        slideWindow.minWindow(s, t);


    }

    /**
     * 字符串中最长无重复最多两个字符子串
     *
     * <a href="http://www.cnblogs.com/grandyang/p/5185561.html">字符串中最长无重复最多两个字符子串</a>
     *
     * @param s
     * @return
     */
    public int lengthOfLongestSubstringTwoDistinct(String s) {
        if (s == null || s.length() == 0) {
            return 0;
        }
        HashMap<Character, Integer> map = new HashMap<>();

        int result = 0;

        int left = 0;
        for (int i = 0; i < s.length(); i++) {
            int count = map.getOrDefault(s.charAt(i), 0);

            map.put(s.charAt(i), ++count);

            while (map.size() > 2) {
                char c = s.charAt(left);
                int num = map.get(c);
                num--;
                if (num == 0) {
                    map.remove(c);
                } else {
                    map.put(c, num);
                }
                left++;
            }

            result = Math.max(result, i - left + 1);
        }
        return result;
    }


    /**
     * 76. Minimum Window Substring
     * trick 滑动窗口思路
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
        for (int i = 0; i < t.length(); i++) {
            hash[t.charAt(i) - '0']++;
        }

        /**
         * 窗口大小维持 为 t.length() 大小
         */
        int count = t.length();

        int result = Integer.MAX_VALUE;

        int begin = 0;

        int end = 0;

        int head = 0;

        // 遍历字符串
        // 遍历过程中移动窗口大小
        while (end < s.length()) {

            // 由于窗口太小 移动窗口右边界
            if (hash[s.charAt(end++) - '0']-- > 0) {
                count--;
            }

            // 窗口大小达到条件时
            // 停止移动窗口
            // 根据窗口当前大小获取条件值
            // 判断如何移动窗口
            while (count == 0) {
                if (end - begin < result) {
                    result = end - begin;
                    head = begin;
                }
                if (hash[s.charAt(begin++) - '0']++ == 0) {
                    count++;
                }
            }
        }

        if (result < Integer.MAX_VALUE) {
            return s.substring(head, head + result);
        }
        return "";


    }


}
