package org.learn.algorithm.leetcode;

import org.apache.tomcat.util.descriptor.web.WebXml;
import org.learn.algorithm.datastructure.Trie;

import javax.swing.table.TableModel;
import java.util.*;

/**
 * @author luk
 * @date 2021/4/7
 */
public class RecursiveSolution {


    public static void main(String[] args) {
        RecursiveSolution solution = new RecursiveSolution();
        char[][] words = new char[][]{{'o', 'a', 'b', 'n'}, {'o', 't', 'a', 'e'}, {'a', 'h', 'k', 'r'}, {'a', 'f', 'l', 'v'}};
        String[] tmp = new String[]{"oa", "oaa"};
//        solution.findWords(words, tmp);
//        List<Integer> diffWaysToCompute = solution.diffWaysToCompute("2*3-4*5");
//        System.out.println(diffWaysToCompute);
//        solution.getFactors(8);
//        solution.combinationSum4(new int[]{1, 2, 3}, 4);
//        solution.combinationSum2(new int[]{3, 1, 3, 5, 1, 1,
//                8}, 8);
//        solution.permuteUnique(new int[]{1, 1, 2});
//        System.out.println(solution.wordBreakII("leetcode", Arrays.asList("leet", "code")));
        String[] wordArray = new String[]{"oath", "pea", "eat", "rain"};

//        List<String> result = solution.findWords(words, wordArray);
//        System.out.println(result);
//        System.out.println(solution.diffWaysToCompute("2-1-1"));
        List<List<Integer>> factors = solution.getFactors(8);
        System.out.println(factors);
    }

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
        int endIndex = nums.length - 1;

        while (endIndex > 0) {
            if (nums[endIndex] > nums[endIndex - 1]) {
                break;
            }
            endIndex--;
        }
        if (endIndex == 0) {
            reverseArrays(nums, 0, nums.length - 1);
            return;
        }
        int j = nums.length - 1;

        while (j > endIndex - 1) {
            if (nums[j] > nums[endIndex - 1]) {
                break;
            }
            j--;
        }
        swap(nums, endIndex - 1, j);

        reverseArrays(nums, endIndex, nums.length - 1);
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
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(candidates);
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

    /**
     * 40 Combination Sum II
     *
     * @param candidates
     * @param target
     * @return
     */
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        if (candidates == null || candidates.length == 0) {
            return new ArrayList<>();
        }
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(candidates);
        internalCombinationSum2(result, new ArrayList<>(), 0, candidates, target);
        return result;
    }

    private void internalCombinationSum2(List<List<Integer>> result, List<Integer> tmp, int start, int[] candidates, int target) {
        if (target == 0) {
            result.add(new ArrayList<>(tmp));
            return;
        }
        for (int i = start; i < candidates.length && candidates[i] <= target; i++) {
            if (i > start && candidates[i] == candidates[i - 1]) {
                continue;
            }
            tmp.add(candidates[i]);
            internalCombinationSum2(result, tmp, i + 1, candidates, target - candidates[i]);
            tmp.remove(tmp.size() - 1);
        }
    }


    /**
     * 46. Permutations
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> permute(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new ArrayList<>();
        }
        boolean[] used = new boolean[nums.length];
        List<List<Integer>> result = new ArrayList<>();
        internalPermute(result, new ArrayList<>(), used, nums);
        return result;
    }

    private void internalPermute(List<List<Integer>> result, List<Integer> tmp, boolean[] used, int[] nums) {
        if (tmp.size() == nums.length) {

            result.add(new ArrayList<>(tmp));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (used[i]) {
                continue;
            }
            used[i] = true;
            tmp.add(nums[i]);
            internalPermute(result, tmp, used, nums);
            used[i] = false;
            tmp.remove(tmp.size() - 1);
        }
    }


    /**
     * 47. Permutations II
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> permuteUnique(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new ArrayList<>();
        }
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);
        boolean[] used = new boolean[nums.length];
        internalPermuteUnique(result, new ArrayList<>(), used, nums);
        return result;
    }

    private void internalPermuteUnique(List<List<Integer>> result, List<Integer> tmp, boolean[] used, int[] nums) {
        if (tmp.size() == nums.length) {
            result.add(new ArrayList<>(tmp));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (i > 0 && nums[i] == nums[i - 1] && !used[i - 1]) {
                continue;
            }
            if (used[i]) {
                continue;
            }
            used[i] = true;
            tmp.add(nums[i]);
            internalPermuteUnique(result, tmp, used, nums);
            used[i] = false;
            tmp.remove(tmp.size() - 1);
        }
    }

    /**
     * 51. N-Queens
     *
     * @param n
     * @return
     */

    public List<List<String>> solveNQueens(int n) {
        if (n <= 0) {
            return new ArrayList<>();
        }
        char[][] matrix = new char[n][n];
        for (char[] row : matrix) {
            Arrays.fill(row, '.');
        }
        List<List<String>> result = new ArrayList<>();
        internalSolveQueens(result, matrix, 0, n);
        return result;
    }

    private void internalSolveQueens(List<List<String>> result, char[][] matrix, int row, int n) {
        if (row == n) {
            List<String> tmp = new ArrayList<>();
            for (char[] queen : matrix) {
                tmp.add(String.valueOf(queen));
            }
            result.add(tmp);
            return;
        }
        for (int j = 0; j < n; j++) {
            if (checkValid(matrix, j, row)) {
                matrix[row][j] = 'Q';
                internalSolveQueens(result, matrix, row + 1, n);
                matrix[row][j] = '.';
            }
        }
    }

    private boolean checkValid(char[][] matrix, int column, int row) {
        for (int i = row - 1; i >= 0; i--) {
            if (matrix[i][column] == 'Q') {
                return false;
            }
        }
        for (int i = row - 1, j = column - 1; i >= 0 && j >= 0; i--, j--) {
            if (matrix[i][j] == 'Q') {
                return false;
            }
        }
        for (int i = row - 1, j = column + 1; i >= 0 && j < matrix[0].length; i--, j++) {
            if (matrix[i][j] == 'Q') {
                return false;
            }
        }
        return true;
    }


    /**
     * 77. Combinations
     *
     * @param n
     * @param k
     * @return
     */
    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> result = new ArrayList<>();
        intervalCombine(result, new ArrayList<>(), 1, n, k);
        return result;
    }

    private void intervalCombine(List<List<Integer>> result, List<Integer> tmp, int start, int n, int k) {
        if (tmp.size() == k) {
            result.add(new ArrayList<>(tmp));
            return;
        }
        for (int i = start; i <= n; i++) {
            tmp.add(i);
            intervalCombine(result, tmp, i + 1, n, k);
            tmp.remove(tmp.size() - 1);
        }
    }

    /**
     * 78. Subsets
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> subsets(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new ArrayList<>();
        }
        List<List<Integer>> result = new ArrayList<>();
        internalSubsets(result, new ArrayList<>(), 0, nums);
        return result;
    }

    private void internalSubsets(List<List<Integer>> result, List<Integer> tmp, int start, int[] nums) {
        result.add(new ArrayList<>(tmp));
        for (int i = start; i < nums.length; i++) {
            tmp.add(nums[i]);
            internalSubsets(result, tmp, i + 1, nums);
            tmp.remove(tmp.size() - 1);
        }
    }

    /**
     * 90. Subsets II
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new ArrayList<>();
        }
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);
        internalSubsetsWithDup(result, new ArrayList<>(), 0, nums);
        return result;
    }

    private void internalSubsetsWithDup(List<List<Integer>> result, List<Integer> tmp, int start, int[] nums) {
        result.add(new ArrayList<>(tmp));
        for (int i = start; i < nums.length; i++) {
            if (i > start && nums[i] == nums[i - 1]) {
                continue;
            }
            tmp.add(nums[i]);
            internalSubsetsWithDup(result, tmp, i + 1, nums);
            tmp.remove(tmp.size() - 1);
        }
    }

    /**
     * todo
     * 60. Permutation Sequence
     *
     * @param n
     * @param k
     * @return
     */
    public String getPermutation(int n, int k) {
        List<Integer> nums = new ArrayList<>();
        for (int i = 1; i <= n; i++) {
            nums.add(i);
        }
        int base = 1;

        return "";
    }


    /**
     * 216. Combination Sum III
     *
     * @param k
     * @param n
     * @return
     */
    public List<List<Integer>> combinationSum3(int k, int n) {
        if (k <= 0 || n <= 0) {
            return new ArrayList<>();
        }
        List<List<Integer>> result = new ArrayList<>();
        internalCombineSum3(result, new ArrayList<>(), 1, n, k, n);
        return result;
    }

    private void internalCombineSum3(List<List<Integer>> result, List<Integer> tmp, int start, int n, int k, int target) {
        if (tmp.size() == k && target == 0) {
            result.add(new ArrayList<>(tmp));
            return;
        }
        for (int i = start; i <= 9 && i <= target; i++) {
            tmp.add(i);
            internalCombineSum3(result, tmp, i + 1, n, k, target - i);
            tmp.remove(tmp.size() - 1);
        }
    }

    /**
     * 377. Combination Sum IV
     *
     * @param nums
     * @param target
     * @return
     */
    public int combinationSum4(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int[] result = new int[target + 1];
        result[0] = 1;

        for (int i = 1; i <= target; i++) {
            for (int j = 0; j < nums.length; j++) {
                if (i - nums[j] >= 0) {
                    result[i] += result[i - nums[j]];
                }
            }
        }
        return result[target];
    }


    /**
     * 254
     * Factor Combinations
     *
     * @param n: a integer
     * @return: return a 2D array
     */
    public List<List<Integer>> getFactors(int n) {
        if (n <= 1) {
            return new ArrayList<>();
        }
        List<List<Integer>> result = new ArrayList<>();
        internalGetFactors(result, new ArrayList<>(), 2, n / 2, 1, n);
        return result;
    }

    private void internalGetFactors(List<List<Integer>> result, List<Integer> tmp, int start, int end, int value, int expectedValue) {
        if (value == expectedValue) {
            result.add(new ArrayList<>(tmp));
            return;
        }
        if (value > expectedValue) {
            return;
        }
        for (int i = start; i <= end && value * i <= expectedValue; i++) {
            if (expectedValue % i != 0) {
                continue;
            }
            tmp.add(i);
            internalGetFactors(result, tmp, i, end, value * i, expectedValue);
            tmp.remove(tmp.size() - 1);
        }
    }


}
