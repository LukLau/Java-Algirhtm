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
        int furthest = 0;
        int currentIndex = 0;
        for (int i = 0; i < nums.length - 1; i++) {
            furthest = Math.max(i + nums[i], furthest);
            if (currentIndex == i) {
                step++;
                currentIndex = furthest;
            }
        }
        return step;
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
