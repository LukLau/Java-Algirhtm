package org.learn.algorithm.leetcode;


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


}
