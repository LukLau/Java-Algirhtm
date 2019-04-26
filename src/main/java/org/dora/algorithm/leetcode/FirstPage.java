package org.dora.algorithm.leetcode;

import org.dora.algorithm.datastructe.ListNode;

import java.util.HashMap;

/**
 * @author dora
 * @date 2019-04-26
 */
public class FirstPage {

    /**
     * 1. Two Sum
     * @param nums
     * @param target
     * @return
     */
    public int[] twoSum(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return new int[]{};
        }
        int[] ans = new int[2];
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(target - nums[i])) {

                ans[0] = map.get(target-nums[i]);
                ans[1] = i;
            }
            map.put(nums[i], i);
        }
        return ans;
    }

    /**
     * 2. Add Two Numbers
     *
     * @param l1
     * @param l2
     * @return
     */
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {

    }
}
