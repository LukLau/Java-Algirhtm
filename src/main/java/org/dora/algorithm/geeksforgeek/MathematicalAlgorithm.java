package org.dora.algorithm.geeksforgeek;

import org.dora.algorithm.datastructe.Node;

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


    // ----------图理论graph----//

    /**
     * 133. Clone Graph
     *
     * @param node
     * @return
     */
    public Node cloneGraph(Node node) {
        return null;
    }


    // -------位运算 Bits -----------//

    /**
     * 136. Single Number
     *
     * @param nums
     * @return
     */
    public int singleNumber(int[] nums) {
        if (nums == null || nums.length == 0) {
            return Integer.MAX_VALUE;
        }
        int result = 0;

        for (int num : nums) {

            result ^= num;
        }
        return result;
    }

    /**
     * https://leetcode.com/problems/single-number-ii/discuss/43295/Detailed-explanation-and-generalization-of-the-bitwise-operation-method-for-single-numbers
     * 137. Single Number II
     *
     * @param nums
     * @return
     */
    public int singleNumberII(int[] nums) {
        return 0;
    }


    /**
     * 149. Max Points on a Line
     *
     * @param points
     * @return
     */
    public int maxPoints(int[][] points) {
        return 0;
    }

    // ----------连续数字最大乘积--//

    /**
     * 152. Maximum Product Subarray
     *
     * @param nums
     * @return
     */
    public int maxProduct(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int preMax = nums[0];
        int preMin = nums[0];
        int result = preMax;
        for (int i = 1; i < nums.length; i++) {
            int tmpMax = Math.max(Math.max(preMax * nums[i], preMin * nums[i]), nums[i]);
            int tmpMin = Math.min(Math.min(preMin * nums[i], preMax * nums[i]), nums[i]);
            result = Math.max(tmpMax, result);
            preMax = tmpMax;
            preMin = tmpMin;
        }
        return result;
    }

    /**
     * 151. Reverse Words in a String
     *
     * @param s
     * @return
     */
    public String reverseWords(String s) {
        if (s == null || s.isEmpty()) {
            return "";
        }
        s = s.trim();
        String[] split = s.split(" ");
        StringBuilder builder = new StringBuilder();
        for (int i = split.length - 1; i >= 0; i--) {
            if (split[i].isEmpty()) {
                continue;
            }
            builder.append(split[i]);
            if (i > 0) {
                builder.append(" ");
            }
        }
        return builder.toString();
    }


}
