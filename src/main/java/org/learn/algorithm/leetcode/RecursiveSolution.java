package org.learn.algorithm.leetcode;

import org.learn.algorithm.datastructure.WordDictionary;

import java.util.*;

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

    public static void main(String[] args) {
        RecursiveSolution solution = new RecursiveSolution();

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
        for (int i = start; i < candidates.length && target >= candidates[i]; i++) {
            if (i > start && candidates[i] == candidates[i - 1]) {
                continue;
            }
            tmp.add(candidates[i]);
            intervalCombinationII(ans, tmp, i + 1, candidates, target - candidates[i]);
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
        List<List<Integer>> ans = new ArrayList<>();
        boolean[] used = new boolean[nums.length];
        intervalPermute(ans, new ArrayList<>(), used, nums);
        return ans;
    }

    private void intervalPermute(List<List<Integer>> ans, List<Integer> tmp, boolean[] used, int[] nums) {
        if (tmp.size() == nums.length) {
            ans.add(new ArrayList<>(tmp));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (used[i]) {
                continue;
            }
            used[i] = true;
            tmp.add(nums[i]);
            intervalPermute(ans, tmp, used, nums);
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
        if (nums == null) {
            return new ArrayList<>();
        }
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);
        boolean[] used = new boolean[nums.length];
        intervalPermuteUnique(result, new ArrayList<>(), used, nums);
        return result;
    }

    private void intervalPermuteUnique(List<List<Integer>> result, List<Integer> tmp, boolean[] used, int[] nums) {
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
            tmp.add(nums[i]);
            used[i] = true;
            intervalPermuteUnique(result, tmp, used, nums);
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
        intervalSubSets(result, new ArrayList<>(), 0, nums);
        return result;

    }

    private void intervalSubSets(List<List<Integer>> result, List<Integer> tmp, int start, int[] nums) {
        result.add(new ArrayList<>(tmp));
        for (int i = start; i < nums.length; i++) {
            tmp.add(nums[i]);
            intervalSubSets(result, tmp, i + 1, nums);
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
        Arrays.sort(nums);
        List<List<Integer>> result = new ArrayList<>();
        intervalUniqueSet(result, new ArrayList<>(), 0, nums);
        return result;
    }

    private void intervalUniqueSet(List<List<Integer>> result, List<Integer> tmp, int start, int[] nums) {
        result.add(new ArrayList<>(tmp));
        for (int i = start; i < nums.length; i++) {
            if (i > start && nums[i] == nums[i - 1]) {
                continue;
            }
            tmp.add(nums[i]);
            intervalUniqueSet(result, tmp, i + 1, nums);
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
        if (n <= 0) {
            return "";
        }
        List<Integer> nums = new ArrayList<>();
        for (int i = 1; i <= n; i++) {
            nums.add(i);
        }
        int[] factors = new int[n + 1];
        factors[0] = 1;
        int base = 1;
        for (int i = 1; i <= n; i++) {
            base *= i;
            factors[i] = i;
        }
        StringBuilder builder = new StringBuilder();
        k--;
        for (int i = 0; i < n; i++) {
            int index = k / factors[n - 1 - i];
            builder.append(index);
            nums.remove(index);
            k -= index * factors[n - 1 - i];
        }
        return builder.toString();
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
        for (int i = start; i <= n; i++) {
            tmp.add(i);
            combineSum3(result, tmp, i + 1, k, n - i);
            tmp.remove(tmp.size() - 1);
        }
    }


    /**
     * 254
     * Factor Combinations
     *
     * @param n: a integer
     * @return: return a 2D array
     */
    public List<List<Integer>> getFactors(int n) {
        if (n <= 0) {
            return new ArrayList<>();
        }
        List<List<Integer>> result = new ArrayList<>();
        // write your code here
        intervalFactors(result, new ArrayList<>(), 2, n);
        return result;
    }

    private void intervalFactors(List<List<Integer>> result, List<Integer> tmp, int start, int n) {
        if (n == 1 && tmp.size() > 1) {
            result.add(new ArrayList<>(tmp));
            return;
        }
        for (int i = start; i <= n; i++) {
            if (n % i != 0) {
                continue;
            }
            tmp.add(i);
            intervalFactors(result, tmp, i, n / i);
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
        if (i < 0 || i >= used.length || j < 0 || j >= used[i].length || used[i][j] || board[i][j] != word.charAt(k)) {
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
        WordDictionary wordDictionary = new WordDictionary();
        for (String word : words) {
            wordDictionary.addWord(word);
        }
        int row = board.length;

        int column = board[0].length;

        boolean[][] used = new boolean[row][column];

        List<String> result = new ArrayList<>();

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                intervalDfs(result, used, i, j, wordDictionary, "", board);
            }
        }
        return result;
    }

    private void intervalDfs(List<String> result, boolean[][] used, int i, int j, WordDictionary wordDictionary, String s, char[][] board) {
        if (i < 0 || i >= used.length || j <= 0 || j >= used[i].length) {
            return;
        }
        s += board[i][j];
        if (!wordDictionary.startWith(s)) {
            return;
        }
        if (wordDictionary.search(s)) {
            result.add(s);
        }
        used[i][j] = true;
        intervalDfs(result, used, i - 1, j, wordDictionary, s, board);
        intervalDfs(result, used, i + 1, j, wordDictionary, s, board);
        intervalDfs(result, used, i, j - 1, wordDictionary, s, board);
        intervalDfs(result, used, i, j + 1, wordDictionary, s, board);
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
                boolean edge = i == 0 || i == row - 1 || j == 0 || j == column - 1;
                if (edge && board[i][j] == 'O') {
                    intervalSolve(i, j, board);
                }
            }
        }
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (board[i][j] == '1') {
                    board[i][j] = 'O';
                } else if (board[i][j] == 'O') {
                    board[i][j] = 'X';
                }
            }
        }
    }

    private void intervalSolve(int i, int j, char[][] board) {
        if (i < 0 || i >= board.length || j < 0 || j >= board[i].length || board[i][j] != 'O') {
            return;
        }
        board[i][j] = '1';
        intervalSolve(i - 1, j, board);
        intervalSolve(i + 1, j, board);
        intervalSolve(i, j - 1, board);
        intervalSolve(i, j + 1, board);
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
        int result = 0;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (grid[i][j] == '1') {
                    intervalIslands(grid, i, j);
                    result++;
                }
            }
        }
        return result;
    }

    private void intervalIslands(char[][] grid, int i, int j) {
        if (i < 0 || i >= grid.length || j < 0 || j >= grid[i].length || grid[i][j] != '1') {
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
                    intervalWallsAndGates(i, j, rooms, 0);
                }
            }
        }
        // write your code here
    }

    private void intervalWallsAndGates(int i, int j, int[][] rooms, int distance) {
        if (i < 0 || i >= rooms.length || j < 0 || j >= rooms[i].length || rooms[i][j] == -1) {
            return;
        }
        if (rooms[i][j] < distance) {
            return;
        }
        if (rooms[i][j] == Integer.MAX_VALUE || rooms[i][j] > distance) {
            rooms[i][j] = distance;
        }
        intervalWallsAndGates(i - 1, j, rooms, distance + 1);
        intervalWallsAndGates(i + 1, j, rooms, distance + 1);
        intervalWallsAndGates(i, j - 1, rooms, distance + 1);
        intervalWallsAndGates(i, j + 1, rooms, distance + 1);
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
        if (s == null || s.isEmpty()) {
            return false;
        }
        Map<String, Boolean> map = new HashMap<>();
        return intervalWordBreak(map, s, wordDict);
    }

    private boolean intervalWordBreak(Map<String, Boolean> map, String s, List<String> wordDict) {
        if (map.containsKey(s)) {
            return map.get(s);
        }
        if (s.isEmpty()) {
            return true;
        }
        for (String word : wordDict) {
            int index = s.indexOf(word);
            if (index != 0) {
                continue;
            }
            if (intervalWordBreak(map, s.substring(word.length()), wordDict)) {
                return true;
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
        return intervalWordBreakII(map, s, wordDict);
    }

    private List<String> intervalWordBreakII(Map<String, List<String>> map, String s, List<String> wordDict) {
        if (map.containsKey(s)) {
            return map.get(s);
        }
        List<String> result = new ArrayList<>();
        if (s.isEmpty()) {
            result.add(s);
            return result;
        }
        for (String word : wordDict) {
            int index = s.indexOf(word);
            if (index != 0) {
                continue;
            }
            List<String> tmpList = intervalWordBreakII(map, s.substring(word.length()), wordDict);
            for (String tmp : tmpList) {
                String s1 = word + (tmp.isEmpty() ? "" : " " + tmp);
                result.add(s1);
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
        if (expression == null || expression.isEmpty()) {
            return new ArrayList<>();
        }
        char[] words = expression.toCharArray();
        List<String> params = new ArrayList<>();
        int endIndex = 0;
        while (endIndex < words.length) {
            if (Character.isDigit(words[endIndex])) {
                int tmp = 0;
                while (endIndex < words.length && Character.isDigit(words[endIndex])) {
                    tmp = tmp * 10 + Character.getNumericValue(words[endIndex++]);
                }
                params.add(String.valueOf(tmp));
            } else {
                params.add(String.valueOf(words[endIndex++]));
            }
        }
        String[] expressions = params.toArray(new String[]{});
        return intervalDiffWays(expressions, 0, expressions.length - 1);

    }

    private List<Integer> intervalDiffWays(String[] words, int start, int end) {
        List<Integer> result = new ArrayList<>();
        if (start == end) {
            result.add(Integer.parseInt(words[start]));
            return result;
        }
        if (start > end) {
            return result;
        }
        for (int i = start + 1; i <= end - 1; i = i + 2) {
            List<Integer> leftNums = intervalDiffWays(words, start, i - 1);
            List<Integer> rightNums = intervalDiffWays(words, i + 1, end);
            String sign = words[i];
            for (Integer leftNum : leftNums) {
                int val = 0;
                for (Integer rightNum : rightNums) {
                    if ("+".equals(sign)) {
                        val = leftNum + rightNum;
                    } else if ("-".equals(sign)) {
                        val = leftNum - rightNum;
                    } else if ("*".equals(sign)) {
                        val = leftNum * rightNum;
                    } else if ("/".equals(sign)) {
                        val = leftNum / rightNum;
                    }
                    result.add(val);
                }
            }
        }
        return result;
    }


    /**
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
        intervalAdd(result, 0, 0, 0, num, target, "");
        return result;
    }

    private void intervalAdd(List<String> result, long val, int pos, long multi, String num, int target, String s) {
        if (pos == num.length() && val == target) {
            result.add(s);
            return;
        }
        for (int i = pos; i < num.length(); i++) {
            if (i > pos && num.charAt(pos) == '0') {
                continue;
            }
            String substring = num.substring(pos, i + 1);
            long parseInt = Long.parseLong(substring);
            if (pos == 0) {
                intervalAdd(result, parseInt, i + 1, parseInt, num, target, s + parseInt);
            } else {
                intervalAdd(result, val + parseInt, i + 1, parseInt, num, target, s + "+" + parseInt);
                intervalAdd(result, val - parseInt, i + 1, -parseInt, num, target, s + "-" + parseInt);
                intervalAdd(result, val - multi + multi * parseInt, i + 1, multi * parseInt, num, target, s + "*" + parseInt);
            }

        }
    }


}
