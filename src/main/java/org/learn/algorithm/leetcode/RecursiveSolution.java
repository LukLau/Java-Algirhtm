package org.learn.algorithm.leetcode;

import java.util.ArrayList;
import java.util.List;

/**
 * @author luk
 * @date 2021/4/7
 */
public class RecursiveSolution {


    /**
     * 22. Generate Parentheses
     *
     * @param n
     * @return
     */
    public List<String> generateParenthesis(int n) {
        if (n <= 0) {
            return new ArrayList<>();
        }
        List<String> result = new ArrayList<>();
        intervalGenerate(result, 0, 0, n, "");
        return result;
    }

    private void intervalGenerate(List<String> result, int open, int close, int n, String s) {
        if (s.length() == 2 * n) {
            result.add(s);
            return;
        }
        if (open < n) {
            intervalGenerate(result, open + 1, close, n, s + "(");
        }
        if (close < open) {
            intervalGenerate(result, open, close + 1, n, s + ")");
        }
    }


    /**
     * 31. Next Permutation
     *
     * @param nums
     */
    public void nextPermutation(int[] nums) {
        if (nums == null || nums.length == 0) {
            return;
        }
        int index = nums.length - 1;
        while (index > 0) {
            if (nums[index] > nums[index - 1]) {
                break;
            }
            index--;
        }
        if (index == 0) {
            reverseArrays(nums,0,nums.length-1);
            return;
        }
        int end = nums.length - 1;
        while (end > index - 1) {
            if (nums[end] > nums[index - 1]) {
                break;
            }
            end--;
        }
        swap(nums, end, index - 1);
        reverseArrays(nums, index, nums.length - 1);
    }

    private void swap(int[] nums, int i, int j) {
        int val = nums[i];
        nums[i] = nums[j];
        nums[j] = val;
    }


    public void reverseArrays(int[] nums, int start, int end) {
        if (start > end) {
            return;
        }
        for (int i = start; i <= (start + end) / 2; i++) {
            int value = nums[i];
            nums[i] = nums[start + end - i];
            nums[start + end - i] = value;
        }
    }

    public static void main(String[] args) {
        RecursiveSolution solution = new RecursiveSolution();
        int[] nums = new int[]{3, 2, 1};
        solution.nextPermutation(nums);
    }

}
