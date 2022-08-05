package org.learn.algorithm.leetcode;

import org.learn.algorithm.datastructure.Trie;

import javax.swing.*;
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
        solution.getFactors(8);
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
        intervalCombinationSum2(result, new ArrayList<>(), candidates, 0, target);
        return result;
    }

    private void intervalCombinationSum2(List<List<Integer>> result, List<Integer> tmp, int[] candidates, int start, int target) {
        if (target == 0) {
            result.add(new ArrayList<>(tmp));
            return;
        }
        for (int i = start; i < candidates.length && candidates[i] <= target; i++) {
            if (i > start && candidates[i] == candidates[i - 1]) {
                continue;
            }
            tmp.add(candidates[i]);
            intervalCombinationSum2(result, tmp, candidates, i + 1, target - candidates[i]);
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
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);
        boolean[] used = new boolean[nums.length];
        internalPermute(result, new ArrayList<>(), nums, used);
        return result;
    }

    private void internalPermute(List<List<Integer>> result, List<Integer> tmp, int[] nums, boolean[] used) {
        if (tmp.size() == nums.length) {
            result.add(new ArrayList<>(tmp));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (used[i]) {
                continue;
            }
            tmp.add(nums[i]);
            used[i] = true;
            internalPermute(result, tmp, nums, used);
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
        internalPermuteUnique(result, new ArrayList<>(), nums, used);
        return result;
    }

    private void internalPermuteUnique(List<List<Integer>> result, List<Integer> tmp, int[] nums, boolean[] used) {
        if (tmp.size() == nums.length) {
            result.add(new ArrayList<>(tmp));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (used[i]) {
                continue;
            }
            if (i > 0 && nums[i] == nums[i - 1] && !used[i - 1]) {
                continue;
            }
            used[i] = true;
            tmp.add(nums[i]);
            internalPermuteUnique(result, tmp, nums, used);
            used[i] = false;
            tmp.remove(tmp.size() - 1);
        }
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
        intervalSubsets(result, new ArrayList<>(), 0, nums);
        return result;
    }

    private void intervalSubsets(List<List<Integer>> result, List<Integer> tmp, int start, int[] nums) {
        result.add(new ArrayList<>(tmp));
        for (int i = start; i < nums.length; i++) {
            tmp.add(nums[i]);
            intervalSubsets(result, tmp, i + 1, nums);
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
        combineSum3(result, new ArrayList<Integer>(), 1, k, n);
        return result;
    }

    private void combineSum3(List<List<Integer>> result, ArrayList<Integer> tmp, int start, int k, int n) {
        if (tmp.size() == k && n == 0) {
            result.add(new ArrayList<>(tmp));
            return;
        }
        for (int i = start; i <= 9 && i <= n; i++) {
            tmp.add(i);
            combineSum3(result, tmp, i + 1, k, n - i);
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
        int[][] dp = new int[target + 1][nums.length];

        for (int i = 1; i <= target; i++) {
            for (int j = 0; j < nums.length; j++) {
                if (i >= nums[j]) {
                    dp[i][j] = 1 + dp[i - nums[j]][j];
                } else {
                    dp[i][j] = dp[i][j - 1];
                }
            }
        }
        return dp[target][nums.length - 1];
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
        intervalGetFactors(result, new ArrayList<>(), 2, n / 2, 1, n);
        return result;
    }

    private void intervalGetFactors(List<List<Integer>> result, List<Integer> tmp, int start, int end, int value, int n) {
        if (value == n) {
            result.add(new ArrayList<>(tmp));
            return;
        }
        for (int i = start; i <= end && value * i <= n; i++) {
            if (n % i != 0) {
                continue;
            }
            tmp.add(i);
            intervalGetFactors(result, tmp, i, end, value * i, n);
            tmp.remove(tmp.size() - 1);
        }
    }

    // --DFS/BFS //


    /**
     * 79. Word Search
     *
     * @param board
     * @param word
     * @return
     */
    public boolean exist(char[][] board, String word) {
        if (board == null || board.length == 0) {
            return false;
        }
        int row = board.length;

        int column = board[0].length;
        boolean[][] used = new boolean[row][column];
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (board[i][j] == word.charAt(0) && validExist(used, i, j, 0, board, word)) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean validExist(boolean[][] used, int i, int j, int k, char[][] board, String word) {
        if (k == word.length()) {
            return true;
        }
        if (i < 0 || i == board.length || j < 0 || j == board[i].length
                || used[i][j] || board[i][j] != word.charAt(k)) {
            return false;
        }
        used[i][j] = true;
        if (validExist(used, i - 1, j, k + 1, board, word) ||
                validExist(used, i + 1, j, k + 1, board, word) ||
                validExist(used, i, j - 1, k + 1, board, word) ||
                validExist(used, i, j + 1, k + 1, board, word)) {
            return true;
        }
        used[i][j] = false;
        return false;

    }


    /**
     * todo
     * 212. Word Search II
     *
     * @param board
     * @param words
     * @return
     */
    public List<String> findWords(char[][] board, String[] words) {
        if (board == null || board.length == 0) {
            return new ArrayList<>();
        }
        List<String> result = new ArrayList<>();
        Trie trie = new Trie();
        for (String word : words) {
            trie.insert(word);
        }
        int row = board.length;
        int column = board[0].length;
        boolean[][] used = new boolean[row][column];
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (trie.startsWith(String.valueOf(board[i][j]))) {
                    internalFindWords(result, trie, i, j, board, used, "");
                }
            }
        }
        return result;

    }

    private void internalFindWords(List<String> result, Trie trie, int i, int j, char[][] board, boolean[][] used, String s) {
        if (i < 0 || i == board.length || j < 0 || j == board[i].length || used[i][j]) {
            return;
        }
        s += board[i][j];

        if (trie.search(s) && !result.contains(s)) {
            result.add(s);
        }
        if (!trie.startsWith(s)) {
            return;
        }
        used[i][j] = true;
        internalFindWords(result, trie, i - 1, j, board, used, s);
        internalFindWords(result, trie, i + 1, j, board, used, s);
        internalFindWords(result, trie, i, j - 1, board, used, s);
        internalFindWords(result, trie, i, j + 1, board, used, s);
        used[i][j] = false;
    }


    /**
     * 130. Surrounded Regions
     *
     * @param board
     */
    public void solve(char[][] board) {
        if (board == null || board.length == 0) {
            return;
        }
        int row = board.length;
        int column = board[0].length;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (board[i][j] == 'O') {
                    internalSurroundedRegion(board, i, j);
                }
            }
        }
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                char tmp = board[i][j];

                if (tmp == 'o') {
                    if (i == 0 || i == row - 1 || j == 0 || j == column - 1) {
                        board[i][j] = 'O';
                    } else {
                        board[i][j] = 'X';
                    }
                }
            }
        }
    }

    private void internalSurroundedRegion(char[][] board, int i, int j) {
        if (i < 0 || i == board.length || j < 0 || j == board[i].length || board[i][j] == 'X' || board[i][j] == 'o') {
            return;
        }
        board[i][j] = 'o';
        internalSurroundedRegion(board, i - 1, j);
        internalSurroundedRegion(board, i + 1, j);
        internalSurroundedRegion(board, i, j - 1);
        internalSurroundedRegion(board, i, j + 1);
    }

    /**
     * 200. Number of Islands
     *
     * @param grid
     * @return
     */
    public int numIslands(char[][] grid) {
        if (grid == null || grid.length == 0) {
            return 0;
        }
        int row = grid.length;
        int column = grid[0].length;
        int count = 0;
        boolean[][] used = new boolean[row][column];
        int[][] matrix = new int[][]{{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        LinkedList<int[]> linkedList = new LinkedList<>();
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (grid[i][j] == '1') {
                    count++;
                    linkedList.offer(new int[]{i, j});
                    while (!linkedList.isEmpty()) {
                        int[] current = linkedList.poll();
                        if (used[current[0]][current[1]]) {
                            continue;
                        }
                        used[current[0]][current[1]] = true;
                        grid[current[0]][current[1]] = '0';
                        for (int[] digits : matrix) {
                            int x = current[0] + digits[0];
                            int y = current[1] + digits[1];
                            if (x < 0 || x >= row || y < 0 || y >= column || grid[x][y] == '0') {
                                continue;
                            }
                            linkedList.offer(new int[]{x, y});
                        }
                    }
                }
            }
        }
        return count;
    }


    /**
     * https://leetcode.cn/problems/parse-lisp-expression/solution/lisp-yu-fa-jie-xi-by-leetcode-solution-zycb/
     * todo
     * 736. Lisp 语法解析
     *
     * @param expression
     * @return
     */
    public int evaluate(String expression) {
        return -1;

    }


    private void intervalIslands(char[][] grid, int i, int j) {
        if (i < 0 || i == grid.length || j < 0 || j == grid[i].length) {
            return;
        }
        if (grid[i][j] == '0') {
            return;
        }
        grid[i][j] = '0';
        intervalIslands(grid, i - 1, j);
        intervalIslands(grid, i + 1, j);
        intervalIslands(grid, i, j - 1);
        intervalIslands(grid, i, j + 1);
    }

    /**
     * 286
     * Walls and Gates
     *
     * @param rooms: m x n 2D grid
     * @return: nothing
     */
    public void wallsAndGates(int[][] rooms) {
        if (rooms == null || rooms.length == 0) {
            return;
        }
        int row = rooms.length;
        int column = rooms[0].length;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (rooms[i][j] == 0) {
                    intervalWalls(rooms, i, j, 0);
                }
            }
        }
    }

    private void intervalWalls(int[][] rooms, int i, int j, int distance) {
        if (i < 0 || i >= rooms.length || j < 0 || j >= rooms[i].length || rooms[i][j] == -1) {
            return;
        }
        if (rooms[i][j] > distance || distance == 0) {
            rooms[i][j] = distance;
            intervalWalls(rooms, i - 1, j, distance + 1);
            intervalWalls(rooms, i + 1, j, distance + 1);
            intervalWalls(rooms, i, j - 1, distance + 1);
            intervalWalls(rooms, i, j + 1, distance + 1);
        }
    }

    // ---- //


    /**
     * todo timeout
     * 87. Scramble String
     *
     * @param s1
     * @param s2
     * @return
     */
    public boolean isScramble(String s1, String s2) {
        if (s1 == null || s2 == null) {
            return false;
        }
        int m = s1.length();
        int n = s2.length();
        if (m != n) {
            return false;
        }
        int[] hash = new int[256];
        for (int i = 0; i < m; i++) {
            hash[s1.charAt(i) - 'a']++;
            hash[s2.charAt(i) - 'a']--;
        }
        for (int num : hash) {
            if (num != 0) {
                return false;
            }
        }
        if (s1.equals(s2)) {
            return true;
        }
        for (int i = 1; i < m; i++) {
            if (isScramble(s1.substring(0, i), s2.substring(0, i)) && isScramble(s1.substring(i), s2.substring(i))) {
                return true;
            }
            if (isScramble(s1.substring(i), s2.substring(0, m - i)) && isScramble(s1.substring(0, i), s2.substring(m - i))) {
                return true;
            }
        }
        return false;
    }


    /**
     * 139. Word Break
     *
     * @param s
     * @param wordDict
     * @return
     */
    public boolean wordBreak(String s, List<String> wordDict) {
        if (s == null || s.isEmpty() || wordDict == null || wordDict.isEmpty()) {
            return false;
        }
        Map<String, Boolean> map = new HashMap<>();
        return internalWordBreak(s, wordDict, map);
    }

    private boolean internalWordBreak(String s, List<String> wordDict, Map<String, Boolean> map) {
        if (s.isEmpty()) {
            return true;
        }
        if (map.containsKey(s)) {
            return map.get(s);
        }
        for (String prefix : wordDict) {
            if (s.startsWith(prefix)) {
                String remain = s.substring(prefix.length());
                if (internalWordBreak(remain, wordDict, map)) {
                    return true;
                }
            }
        }
        map.put(s, false);
        return false;
    }

    /**
     * 140. Word Break II
     *
     * @param s
     * @param wordDict
     * @return
     */
    public List<String> wordBreakII(String s, List<String> wordDict) {
        if (s == null || s.isEmpty()) {
            return new ArrayList<>();
        }
        Map<String, List<String>> map = new HashMap<>();
        return intervalWordBreakII(s, wordDict, map);
    }

    private List<String> intervalWordBreakII(String s, List<String> wordDict, Map<String, List<String>> map) {
        List<String> result = new ArrayList<>();
        if (s.isEmpty()) {
            result.add("");
            return result;
        }
        if (map.containsKey(s)) {
            return map.get(s);
        }

        for (String prefix : wordDict) {
            if (s.startsWith(prefix)) {
                String remain = s.substring(prefix.length());
                List<String> breakWords = intervalWordBreakII(remain, wordDict, map);
                for (String breakWord : breakWords) {
                    String combineWord = prefix + (breakWord.isEmpty() ? "" : " " + breakWord);
                    result.add(combineWord);
                }
            }
        }
        map.put(s, result);
        return result;
    }

    /**
     * 241. Different Ways to Add Parentheses
     *
     * @param expression
     * @return
     */
    public List<Integer> diffWaysToCompute(String expression) {
        if (expression == null || expression.length() == 0) {
            return new ArrayList<>();
        }
        List<String> words = new ArrayList<>();
        char[] expressions = expression.toCharArray();
        int index = 0;
        while (index < expressions.length) {
            if (Character.isDigit(expressions[index])) {
                int tmp = 0;
                while (index < expressions.length && Character.isDigit(expressions[index])) {
                    tmp = tmp * 10 + Character.getNumericValue(expressions[index++]);
                }
                words.add(String.valueOf(tmp));
            }
            if (index < expressions.length && expressions[index] != ' ') {
                words.add(String.valueOf(expressions[index]));
            }
            index++;
        }
        return intervalDiff(0, words.size() - 1, words);
    }

    private List<Integer> intervalDiff(int start, int end, List<String> params) {
        List<Integer> result = new ArrayList<>();
        if (start > end) {
            return result;
        }
        if (start == end) {
            result.add(Integer.parseInt(params.get(start)));
            return result;
        }
        for (int i = start + 1; i <= end - 1; i = i + 2) {
            List<Integer> leftNums = intervalDiff(start, i - 1, params);
            List<Integer> rightNums = intervalDiff(i + 1, end, params);
            String sign = params.get(i);
            for (Integer leftNum : leftNums) {
                for (Integer rightNum : rightNums) {
                    if ("+".equals(sign)) {
                        result.add(leftNum + rightNum);
                    } else if ("-".equals(sign)) {
                        result.add(leftNum - rightNum);
                    } else if ("*".equals(sign)) {
                        result.add(leftNum * rightNum);
                    } else if ("/".equals(sign)) {
                        result.add(leftNum / rightNum);
                    }
                }
            }
        }
        return result;
    }


    /**
     * todo
     * 282. Expression Add Operators
     *
     * @param num
     * @param target
     * @return
     */
    public List<String> addOperators(String num, int target) {
        if (num == null || num.isEmpty()) {
            return new ArrayList<>();
        }
        List<String> result = new ArrayList<>();
        intervalAddOperators(result, 0, 0, "", num, 0, target);
        return result;
    }

    private void intervalAddOperators(List<String> result, int value, int start, String express, String num, int multi, int target) {
        if (start == num.length() && value == target) {
            result.add(express);
            return;
        }
        for (int i = start; i < num.length(); i++) {
            if (i > start && num.charAt(start) == '0') {
                continue;
            }
            String tmp = num.substring(start, i + 1);
            int v = Integer.parseInt(tmp);
            if (start == 0) {
                intervalAddOperators(result, target + value, i + 1, express + tmp, num, v, target);
            } else {
                intervalAddOperators(result, target + value, i + 1, express + "+" + tmp, num, v, target);
                intervalAddOperators(result, target - value, i + 1, express + "-" + tmp, num, -v, target);
                intervalAddOperators(result, target * value, i + 1, express + "*" + tmp, num, -v, target);
            }
        }
    }
}
