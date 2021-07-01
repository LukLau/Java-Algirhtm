package org.learn.algorithm.leetcode;

/**
 * 贪心算法问题
 *
 * @author luk
 * @date 2021/4/8
 */
public class GreedySolution {


    // --跳跃游戏 //

    /**
     * 45. Jump Game II
     *
     * @param nums
     * @return
     */
    public int jumpII(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int step = 0;

        int currentIndex = 0;

        int furthestIndex = nums[0];

        for (int i = 0; i < nums.length - 1; i++) {
            furthestIndex = Math.max(furthestIndex, i + nums[i]);

            if (i == currentIndex) {
                step++;
                currentIndex = furthestIndex;
            }
        }
        return step;
    }

    public boolean canJump(int[] nums) {
        if (nums == null || nums.length == 0) {
            return false;
        }
        int reach = 0;

        for (int i = 0; i < nums.length - 1 && i <= reach; i++) {
            reach = Math.max(reach, i + nums[i]);
        }
        return reach >= nums.length - 1;
    }


    // 卖煤气问题//

    /**
     * 134. Gas Station
     *
     * @param gas
     * @param cost
     * @return
     */
    public int canCompleteCircuit(int[] gas, int[] cost) {
        if (gas == null || cost == null) {
            return -1;
        }
        int global = 0;
        int remain = 0;
        int index = 0;
        for (int i = 0; i < gas.length; i++) {
            global += gas[i] - cost[i];
            remain += gas[i] - cost[i];
            if (remain < 0) {
                index = i + 1;
                remain = 0;
            }
        }
        return global >= 0 ? index : -1;
    }
}
