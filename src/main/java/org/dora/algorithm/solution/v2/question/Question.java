package org.dora.algorithm.solution.v2.question;

/**
 * @author dora
 * @date 2019/10/23
 */
public class Question {

    public static void main(String[] args) {
        int[] nums1 = new int[]{1};
        int[] nums2 = new int[]{2, 3};
        Question question = new Question();
        question.findMedianSortedArrays(nums1, nums2);
    }

    /**
     * 4. Median of Two Sorted Arrays
     *
     * @param nums1
     * @param nums2
     * @return
     */
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int m = nums1.length;
        int n = nums2.length;
        if (n < m) {
            return this.findMedianSortedArrays(nums2, nums1);
        }
        int imin = 0;
        int imax = m;
        int maxLeft = 0;
        int minRight = 0;

        boolean odd = ((m + n) % 2 == 1);
        while (imin <= imax) {
            int i = (imin + imax) / 2;
            int j = (m + n) / 2 - i;
            if (i < m && nums1[i] < nums2[j - 1]) {
                imin = i + 1;
            } else if (i > 0 && nums1[i - 1] > nums2[j]) {
                imax = i - 1;
            } else {
                if (i == 0) {
                    maxLeft = nums2[j - 1];
                } else if (i == m) {
                    maxLeft = nums1[i - 1];
                } else {
                    maxLeft = Math.max(nums1[i - 1], nums2[j - 1]);
                }

                if (odd) {
                    return maxLeft;
                }
                if (i == m) {
                    minRight = nums2[j];
                } else if (j == n) {
                    minRight = nums1[i];
                } else {
                    minRight = Math.min(nums1[i], nums2[j]);
                }
                return (maxLeft + minRight) / 2.0;
            }
        }
        return -1;
    }

    /**
     * 6. ZigZag Conversion
     *
     * @param s
     * @return
     */
    public String longestPalindrome(String s) {
        if (s == null || s.isEmpty()) {
            return "";
        }
        int n = s.length();
        int begin = 0;
        int result = 0;
        boolean[][] dp = new boolean[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                if (s.charAt(j) == s.charAt(i)) {
                    if (i - j < 2) {
                        dp[j][i] = true;
                    } else {
                        dp[j][i] = dp[j + 1][i - 1];
                    }
                }
                if (dp[j][i] && i - j + 1 > result) {

                    begin = j;

                    result = i - j + 1;
                }
            }
        }
        if (result != 0) {
            return s.substring(begin, begin + result);
        }
        return s;
    }

    /**
     * @param s
     * @param numRows
     * @return
     */
    public String convert(String s, int numRows) {
        if (s == null || s.isEmpty()) {
            return "";
        }
        StringBuilder[] builders = new StringBuilder[numRows];
        for (int i = 0; i < builders.length; i++) {
            builders[i] = new StringBuilder();
        }
        char[] chars = s.toCharArray();
        int index = 0;
        while (index < chars.length) {
            for (int i = 0; i < numRows && index < chars.length; i++) {
                builders[i].append(chars[index++]);
            }
            for (int i = numRows - 2; i >= 1 && index < chars.length; i--) {
                builders[i].append(chars[index++]);
            }
        }
        for (int i = 1; i < numRows; i++) {
            builders[0].append(builders[i]);
        }
        return builders[0].toString();
    }
}
