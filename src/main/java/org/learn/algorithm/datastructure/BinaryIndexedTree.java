package org.learn.algorithm.datastructure;

import java.util.*;

/**
 * 树状数组 二叉索引树
 */
public class BinaryIndexedTree {

    /**
     * todo
     * https://www.lintcode.com/problem/1297/description
     *
     * @param nums: a list of integers
     * @return: return a list of integers
     */
    public List<Integer> countSmaller(int[] nums) {
        // write your code here
        if (nums == null || nums.length == 0) {
            return new ArrayList<>();
        }
        List<Integer> result = new ArrayList<>();
        int[] originArray = getOriginArray(nums);
        int[] tree = new int[originArray.length + 5];
        for (int i = nums.length - 1; i >= 0; i--) {
            int index = getIndex(nums, nums[i]);
            result.add(query(tree, index - 1));
            update(tree, index);
        }
        Collections.reverse(result);
        return result;

    }


    private int lowBit(int value) {
        return value & (-value);
    }

    private int query(int[] tree, int position) {
        int ret = 0;
        while (position > 0) {
            ret += tree[position];
            position -= lowBit(position);
        }
        return ret;
    }

    private void update(int[] tree, int position) {
        while (position < tree.length) {
            tree[position] += 1;
            position += lowBit(position);
        }
    }

    private int[] getOriginArray(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int num : nums) {
            set.add(num);
        }
        int[] tmp = new int[set.size()];
        int index = 0;
        for (Integer num : set) {
            tmp[index++] = num;
        }
        Arrays.sort(tmp);
        return tmp;
    }

    private int getIndex(int[] nums, int target) {
        return Arrays.binarySearch(nums, target) + 1;

    }


}
