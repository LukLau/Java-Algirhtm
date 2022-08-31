package org.learn.algorithm.leetcode;

import javax.naming.ldap.PagedResultsResponseControl;
import java.util.Arrays;

/**
 * 排序解决方案
 *
 * @author luk
 * @date 2021/4/13
 */
public class VipSort {

    public static void main(String[] args) {
        VipSort vipSort = new VipSort();
        int[] nums = new int[]{1, 3, 100};
        vipSort.maximumGap(nums);
    }

    // 经典排序//


    // 桶排序

    /**
     * todo
     * 164. Maximum Gap
     */
    public int maximumGap(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        if (nums.length < 2) return 0;
        if (nums.length == 2) {
            return Math.abs(nums[0] - nums[1]);
        }
        int min = Integer.MAX_VALUE;
        int max = Integer.MIN_VALUE;

        for (int num : nums) {
            if (num < min) {
                min = num;
            }
            if (num > max) {
                max = num;
            }
        }
        int bucketSide = (max - min) / nums.length + 1;
        int bucketCount = (max - min) / bucketSide + 1;

        int[] bucketMax = new int[bucketCount];
        int[] bucketMin = new int[bucketCount];

        Arrays.fill(bucketMax, Integer.MIN_VALUE);
        Arrays.fill(bucketMin, Integer.MAX_VALUE);
        for (int num : nums) {
            int index = (num - min) / bucketSide;
            bucketMax[index] = Math.max(bucketMax[index], num);
            bucketMin[index] = Math.min(bucketMin[index], num);
        }
        int result = bucketMax[0] - bucketMin[0];

        int prev = bucketMax[0];

        for (int i = 1; i < bucketMax.length; i++) {
            if (bucketMax[i] == Integer.MIN_VALUE && bucketMax[i] == Integer.MIN_VALUE) {
                continue;
            }
            result = Math.max(result, bucketMin[i] - prev);

            prev = bucketMax[i];
        }
        return result;
    }

}
