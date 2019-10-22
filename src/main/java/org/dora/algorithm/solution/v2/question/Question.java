package org.dora.algorithm.solution.v2.question;

/**
 * @author dora
 * @date 2019/10/23
 */
public class Question {

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

        while (imin <= imax) {
            int i = imin + (imax - imin) / 2;

            int j = (m + n + 1) / 2 - i;

            if (i < m && nums1[i] < nums2[j - 1]) {
                imin = i + 1;
            } else if (i > 0 && nums1[i - 1] > nums2[j]) {
                imax = i - 1;
            } else {
                if (i == 0) {
                    maxLeft = nums2[j - 1];
                } else if (j == 0) {
                    maxLeft = nums1[i - 1];
                } else {
                    maxLeft = Math.max(nums1[i - 1], nums2[j - 1]);
                }

                if ((m + n) % 2 == 1) {
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
}
