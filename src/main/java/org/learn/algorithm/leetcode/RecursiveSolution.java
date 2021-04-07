package org.learn.algorithm.leetcode;

import java.util.ArrayList;
import java.util.Arrays;
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
            reverseArrays(nums, 0, nums.length - 1);
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
        int[] nums = new int[]{2, 3, 6, 7};
        solution.combinationSum(nums, 7);
    }

    // --组合系列问题-//

    /**
     * 39. Combination Sum
     *
     * @param candidates
     * @param target
     * @return
     */
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        if (candidates == null || candidates.length == 0) {
            return new ArrayList<>();
        }
        Arrays.sort(candidates);
        List<List<Integer>> result = new ArrayList<>();
        intervalCombination(result, new ArrayList<>(), 0, candidates, target);
        return result;
    }

    private void intervalCombination(List<List<Integer>> result, List<Integer> tmp, int start, int[] candidates, int target) {
        if (target == 0) {
            result.add(new ArrayList<>(tmp));
            return;
        }
        for (int i = start; i < candidates.length && candidates[i] <= target; i++) {
            tmp.add(candidates[i]);
            intervalCombination(result, tmp, i, candidates, target - candidates[i]);
            tmp.remove(tmp.size() - 1);
        }
    }


    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        if (candidates == null || candidates.length == 0) {
            return new ArrayList<>();
        }
        List<List<Integer>> ans = new ArrayList<>();
        Arrays.sort(candidates);
        intervalCombinationII(ans, new ArrayList<>(), 0, candidates, target);
        return ans;
    }

    private void intervalCombinationII(List<List<Integer>> ans, List<Integer> tmp, int start, int[] candidates, int target) {
        if (target == 0) {
            ans.add(new ArrayList<>(tmp));
            return;
        }
        for (int i = start; i < candidates.length && candidates[i] <= target; i++) {
            if (i > start && candidates[i] == candidates[i - 1]) {
                continue;
            }
            tmp.add(candidates[i]);
            intervalCombinationII(ans, tmp, i + 1, candidates, target - candidates[i]);
            tmp.remove(tmp.size() - 1);
        }
    }


}
