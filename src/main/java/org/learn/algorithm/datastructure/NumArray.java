package org.learn.algorithm.datastructure;

import com.sun.tools.javac.util.Pair;

import java.util.HashMap;
import java.util.Map;

/**
 * todo 使用本地缓存的思想
 * 303. Range Sum Query - Immutable
 *
 * @author luk
 * @date 2021/4/26
 */
public class NumArray {

    private int[] sum;

    public NumArray(int[] nums) {
        sum = new int[nums.length + 1];
        for (int i = 0; i < nums.length; i++) {
            sum[i + 1] = sum[i] + nums[i];
        }

    }

    public int sumRange(int left, int right) {
        return sum[right + 1] - sum[left];
    }
}
