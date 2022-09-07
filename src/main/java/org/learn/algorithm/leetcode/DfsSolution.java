package org.learn.algorithm.leetcode;


import org.learn.algorithm.datastructure.Trie;

import java.util.*;

public class DfsSolution {

    public static void main(String[] args) {
        char[][] board = {{'A', 'B', 'C', 'E'}, {'S', 'F', 'C', 'S'}, {'A', 'D', 'E', 'E'}};
        String word = "ABCCED";
        DfsSolution dfsSolution = new DfsSolution();
        dfsSolution.exist(board, word);
    }


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
                if (board[i][j] == word.charAt(0) && checkValidExist(used, i, j, board, 0, word)) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean checkValidExist(boolean[][] used, int i, int j, char[][] board, int start, String word) {
        if (start >= word.length()) {
            return true;
        }
        if (i < 0 || i == board.length || j < 0 || j == board[i].length || used[i][j]) {
            return false;
        }
        if (board[i][j] != word.charAt(start)) {
            return false;
        }
        used[i][j] = true;
        if (checkValidExist(used, i - 1, j, board, start + 1, word) ||
                checkValidExist(used, i + 1, j, board, start + 1, word) ||
                checkValidExist(used, i, j - 1, board, start + 1, word) ||
                checkValidExist(used, i, j + 1, board, start + 1, word)) {
            return true;
        }
        used[i][j] = false;
        return false;
    }


    /**
     * 130. Surrounded Regions
     *
     * @param board
     */
    public void solveSurroundedRegions(char[][] board) {
        if (board == null || board.length == 0) {
            return;
        }
        int row = board.length;
        int column = board[0].length;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (board[i][j] == 'O' && (i == 0 || i == row - 1 || j == 0 || j == column - 1)) {
                    internalSurrounded(i, j, board);
                }
            }
        }
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (board[i][j] == 'o') {
                    board[i][j] = 'O';
                } else if (board[i][j] == 'O') {
                    board[i][j] = 'X';
                }
            }
        }
    }

    private void internalSurrounded(int i, int j, char[][] board) {
        if (i < 0 || i == board.length || j < 0 || j == board[i].length) {
            return;
        }
        if (board[i][j] != 'O') {
            return;
        }
        board[i][j] = 'o';
        internalSurrounded(i - 1, j, board);
        internalSurrounded(i + 1, j, board);
        internalSurrounded(i, j - 1, board);
        internalSurrounded(i, j + 1, board);
    }


    private void internalNumIsLand(int i, int j, boolean[][] used, char[][] grid) {
        if (i < 0 || i == grid.length || j < 0 || j == grid[i].length || used[i][j]) {
            return;
        }
        used[i][j] = true;
        if (grid[i][j] != '1') {
            return;
        }
        internalNumIsLand(i - 1, j, used, grid);
        internalNumIsLand(i + 1, j, used, grid);

        internalNumIsLand(i, j - 1, used, grid);
        internalNumIsLand(i, j + 1, used, grid);

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
        Trie wordDictionary = new Trie();
        for (String word : words) {
            wordDictionary.insert(word);
        }
        Map<String, Boolean> prefixMap = new HashMap<>();
        Set<String> result = new HashSet<>();
        int row = board.length;
        int column = board[0].length;
        boolean[][] used = new boolean[row][column];
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (wordDictionary.startsWith(String.valueOf(board[i][j]))) {
                    internalFindWords(prefixMap, used, wordDictionary, result, "", i, j, board);
                }
            }
        }
        return new ArrayList<>(result);
    }

    private void internalFindWords(Map<String, Boolean> prefixMap, boolean[][] used, Trie trie, Set<String> result, String s, int i, int j, char[][] board) {
        if (i < 0 || i == board.length || j < 0 || j == board[i].length) {
            return;
        }
        if (used[i][j]) {
            return;
        }
        s += board[i][j];

        if (prefixMap.containsKey(s)) {
            Boolean isStartWord = prefixMap.get(s);
            if (!isStartWord) {
                return;
            }
        }
        if (!trie.startsWith(s)) {
            prefixMap.put(s, false);
            return;
        }
        if (trie.search(s)) {
            prefixMap.put(s, true);
            result.add(s);
        }
        used[i][j] = true;
        internalFindWords(prefixMap, used, trie, result, s, i - 1, j, board);
        internalFindWords(prefixMap, used, trie, result, s, i + 1, j, board);
        internalFindWords(prefixMap, used, trie, result, s, i, j - 1, board);
        internalFindWords(prefixMap, used, trie, result, s, i, j + 1, board);
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


    /**
     * #282 Expression Add Operators
     *
     * @param num:    a string contains only digits 0-9
     * @param target: An integer
     * @return: return all possibilities
     * we will sort your return value in output
     */
    public List<String> addOperatorsii(String num, int target) {
        // write your code here
        if (num == null || num.isEmpty()) {
            return new ArrayList<>();
        }
        List<String> result = new ArrayList<>();
        internalAddOperators(result, "", num, "", 0, 0, target);
//        Collections.sort(result);
        return result;
    }

    private void internalAddOperators(List<String> result, String expression, String words, String sign, long val, long previous, long target) {
        if (words.isEmpty()) {
            return;
        }

        for (int i = 0; i < words.length(); i++) {
            String substring = words.substring(0, i + 1);
            if (substring.length() >= 2 && substring.charAt(0) == '0') {
                continue;
            }
            long parseValue = Long.parseLong(substring);

            String remain = words.substring(i + 1);

            long tmpValue;
            switch (sign) {
                case "+":
                    tmpValue = val + parseValue;
                    break;
                case "-":
                    tmpValue = val - parseValue;
                    parseValue = -parseValue;
                    break;
                case "*":
                    tmpValue = val - previous + previous * parseValue;
                    break;
                default:
                    tmpValue = parseValue;
                    break;
            }
            String tmp = expression + sign + substring;

            if (val == target && remain.isEmpty()) {
                result.add(tmp);
            }
            long multi;
            if (sign.equals("*")) {
                multi = previous * parseValue;
            } else {
                multi = parseValue;
            }
            internalAddOperators(result, tmp, remain, "*", tmpValue, multi, target);
            internalAddOperators(result, tmp, remain, "+", tmpValue, parseValue, target);
            internalAddOperators(result, tmp, remain, "-", tmpValue, parseValue, target);
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
        if (s == null || s.isEmpty()) {
            return false;
        }
        Map<String, Boolean> map = new HashMap<>();
        return internalWordBreak(map, s, wordDict);
    }

    private boolean internalWordBreak(Map<String, Boolean> map, String s, List<String> wordDict) {
        if (s.isEmpty()) {
            return true;
        }
        if (map.containsKey(s)) {
            return map.get(s);
        }
        for (String word : wordDict) {
            if (s.startsWith(word)) {
                String substring = s.substring(word.length());
                if (internalWordBreak(map, substring, wordDict)) {
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
        return internalWordBreakII(s, wordDict);
    }

    private List<String> internalWordBreakII(String s, List<String> wordDict) {
        List<String> result = new ArrayList<>();
        if (s.isEmpty()) {
            result.add(s);
            return result;
        }
        for (String word : wordDict) {
            if (s.startsWith(word)) {
                String substring = s.substring(word.length());

                List<String> wordBreakList = internalWordBreakII(substring, wordDict);

                for (String suffix : wordBreakList) {

                    String tmp = word + (suffix.isEmpty() ? "" : " " + suffix);

                    result.add(tmp);

                }
            }
        }
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
        List<String> wordExpress = new ArrayList<>();
        int startIndex = 0;
        char[] words = expression.toCharArray();
        while (startIndex < words.length) {
            if (Character.isDigit(words[startIndex])) {
                int tmp = 0;
                while (startIndex < words.length && Character.isDigit(words[startIndex])) {
                    tmp = tmp * 10 + Character.getNumericValue(words[startIndex]);
                    startIndex++;
                }
                wordExpress.add(String.valueOf(tmp));
            }
            if (startIndex != words.length && !Character.isDigit(words[startIndex])) {
                wordExpress.add(String.valueOf(words[startIndex]));
            }
            startIndex++;
        }
        return internalDiffwaysToCompute(wordExpress, 0, wordExpress.size() - 1);
    }

    private List<Integer> internalDiffwaysToCompute(List<String> wordExpress, int start, int end) {
        List<Integer> result = new ArrayList<>();
        if (start == end) {
            result.add(Integer.valueOf(wordExpress.get(start)));
            return result;
        }
        for (int i = start + 1; i <= end - 1; i = i + 2) {
            List<Integer> leftNums = internalDiffwaysToCompute(wordExpress, start, i - 1);
            List<Integer> rightNums = internalDiffwaysToCompute(wordExpress, i + 1, end);

            String sign = wordExpress.get(i);

            for (Integer leftNum : leftNums) {
                for (Integer rightNum : rightNums) {
                    if (sign.equals("+")) {
                        result.add(leftNum + rightNum);
                    } else if (sign.equals("-")) {
                        result.add(leftNum - rightNum);
                    } else if (sign.equals("*")) {
                        result.add(leftNum * rightNum);
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
