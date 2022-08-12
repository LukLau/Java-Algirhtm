package org.learn.algorithm.leetcode;

import javax.swing.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class VipRecursive {

    public static void main(String[] args) {
        VipRecursive vipRecursive = new VipRecursive();
//        System.out.println(vipRecursive.addOperators("2320",
//                8));
        int[][] rooms = new int[][]{{2147483647, -1, 0, 2147483647}, {2147483647, 2147483647, 2147483647, -1}, {2147483647, -1, 2147483647, -1}, {0, -1, 2147483647, 2147483647}};

        vipRecursive.wallsAndGates(rooms);
    }

    // DSF //


    /**
     * 286 Walls and Gates
     *
     * @param rooms: m x n 2D grid
     * @return: nothing
     */
    public void wallsAndGates(int[][] rooms) {
        // write your code here
        if (rooms == null || rooms.length == 0) {
            return;
        }
        int row = rooms.length;
        int column = rooms[0].length;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (rooms[i][j] == 0) {
                    internalWallsAndGates(0, i, j, rooms);
                }
            }
        }
    }

    private void internalWallsAndGates(int value, int i, int j, int[][] rooms) {
        if (i < 0 || i == rooms.length || j < 0 || j == rooms[i].length) {
            return;
        }
        if (rooms[i][j] == -1) {
            return;
        }
//        if (rooms[i][j] > value || value == 0) {
        if (value == 0 || rooms[i][j] > value) {
            rooms[i][j] = value;
            internalWallsAndGates(value + 1, i - 1, j, rooms);
            internalWallsAndGates(value + 1, i + 1, j, rooms);
            internalWallsAndGates(value + 1, i, j - 1, rooms);
            internalWallsAndGates(value + 1, i, j + 1, rooms);
        }
    }


}
