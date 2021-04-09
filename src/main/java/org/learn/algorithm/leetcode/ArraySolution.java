package org.learn.algorithm.leetcode;

import com.sun.corba.se.impl.oa.poa.POAPolicyMediatorImpl_NR_USM;

/**
 * 数组系列问题
 *
 * @author luk
 * @date 2021/4/9
 */
public class ArraySolution {


    /**
     * 80. Remove Duplicates from Sorted Array II
     *
     * @param nums
     * @return
     */
    public int removeDuplicates(int[] nums) {
        if (nums == null || nums.length == 0) {
            return -1;
        }
        int count = 1;
        int index = 1;
        for (int i = 1; i < nums.length ; i++) {
            if (nums[i] == nums[i-1]) {
                count++;
                if (count == 3) {
                    continue;
                }
            } else {
                count = 1;
            }
            nums[index++] = nums[i];
        }
        return index;

    }

}
