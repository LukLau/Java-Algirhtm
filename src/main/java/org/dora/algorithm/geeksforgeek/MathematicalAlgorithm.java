package org.dora.algorithm.geeksforgeek;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * @author dora
 * @date 2019/11/4
 */
public class MathematicalAlgorithm {

    /**
     * 牛顿平方法来解决
     * 69. Sqrt(x)
     *
     * @param x
     * @return
     */
    public int mySqrt(int x) {
        double precision = 0.00001;

        double result = x;

        while ((result * result - x) > precision) {
            result = (result + x / result) / 2;
        }
        return (int) result;
    }


    /**
     * todo 牵扯到格雷码 相关知识
     * 89. Gray Code
     * 格雷码
     *
     * @param n
     * @return
     */
    public List<Integer> grayCode(int n) {
        if (n <= 0) {
            return new ArrayList<>();
        }
        for (int i = 0; i < n; i++) {

        }
        return null;
    }


    /**
     * 91. Decode Ways
     *
     * @param s
     * @return
     */
    public int numDecodings(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        }
        return 0;
    }


    /**
     * 93. Restore IP Addresses
     *
     * @param s
     * @return
     */
    public List<String> restoreIpAddresses(String s) {
        if (s == null || s.length() == 0) {
            return new ArrayList<>();
        }
        List<String> ans = new ArrayList<>();
        for (int a = 1; a <= 4; a++) {
            for (int b = 1; b <= 4; b++) {
                for (int c = 1; c <= 4; c++) {
                    for (int d = 1; d <= 4; d++) {
                        int len = a + b + c + d;
                        if (len == 12) {
                            int value1 = Integer.parseInt(s.substring(0, a));
                            int value2 = Integer.parseInt(s.substring(a, a + b));
                            int value3 = Integer.parseInt(s.substring(a + b, a + b + c));
                        }
                    }

                }

            }

        }
        return ans;
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
        int result = 0;
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            if (map.containsKey(num)) {
                continue;
            }
            int leftEdge = map.getOrDefault(num - 1, 0);
            int rightEdge = map.getOrDefault(num + 1, 0);

            int val = leftEdge + rightEdge + 1;

            result = Math.max(result, val);

            map.put(num - leftEdge, val);
            map.put(num + rightEdge, val);
            map.put(num, val);
        }
        return result;
    }
}
