package org.learn.algorithm.leetcode;

import java.util.ArrayList;
import java.util.List;

/**
 * 线段树问题
 *
 * @author luk
 * @date 2021/5/9
 */
public class SegmentTreeSolution {

    /**
     * todo 线段树
     * 315. Count of Smaller Numbers After Self
     *
     * @param nums
     * @return
     */
    public List<Integer> countSmaller(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new ArrayList<>();
        }
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < nums.length; i++) {
            int smallCount = 0;
            int val = nums[i];
            for (int j = i + 1; j < nums.length; j++) {
                if (nums[j] < val) {
                    smallCount++;
                }
            }
            result.add(smallCount);
        }
        return result;
    }



}
