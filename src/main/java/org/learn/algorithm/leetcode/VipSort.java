package org.learn.algorithm.leetcode;

import java.util.Arrays;

/**
 * 排序解决方案
 *
 * @author luk
 * @date 2021/4/13
 */
public class VipSort {

    // 经典排序//


    // 桶排序

    /**
     * todo
     * 164. Maximum Gap
     */
    public int maximumGap(int[] nums) {
        if (nums == null || nums.length <= 1) {
            return 0;
        }
        int min = nums[0];
        int max = nums[0];
        for (int num : nums) {
            if (num < min) {
                min = num;
            }
            if (num > max) {
                max = num;
            }
        }
        if (min == max) {
            return 0;
        }
        int gap = (int) Math.ceil((double) (max - min) / (nums.length - 1));
        int[] minBucket = new int[nums.length];
        int[] maxBucket = new int[nums.length];
        Arrays.fill(minBucket, Integer.MAX_VALUE);
        Arrays.fill(maxBucket, Integer.MIN_VALUE);
        for (int num : nums) {
            int idx = (num - min) / gap;
            minBucket[idx] = Math.min(num, minBucket[idx]);
            maxBucket[idx] = Math.max(num, maxBucket[idx]);
        }
        return -1;
    }

}
