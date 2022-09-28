package org.learn.algorithm.leetcode;

import org.learn.algorithm.datastructure.Point;
import org.learn.algorithm.datastructure.UnionFind;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author luk
 */
public class UnionFindSolution {

    public static void main(String[] args) {
        UnionFindSolution unionFindSolution = new UnionFindSolution();

        List<Point> points = new ArrayList<>();

        Point item1 = new Point(1, 1);
        Point item2 = new Point(0, 1);
        Point item3 = new Point(3, 3);
        Point item4 = new Point(3, 4);

        unionFindSolution.numIslands2(4, 5, new Point[]{item1, item2, item3, item4});
    }


    /**
     * todo
     * NC95 数组中的最长连续子序列
     * max increasing subsequence
     *
     * @param arr int整型一维数组 the array
     * @return int整型
     */
    public int MLS(int[] arr) {
        // write code here
        return -1;
    }

    /**
     * todo 并查集
     * https://www.lintcode.com/problem/434/
     *
     * @param n:         An integer
     * @param m:         An integer
     * @param operators: an array of point
     * @return: an integer array
     */
    public List<Integer> numIslands2(int n, int m, Point[] operators) {
        // write your code here
        List<Integer> result = new ArrayList<>();
        if (n == 0 || m == 0 || operators == null || operators.length == 0) {
            return result;
        }
        int[][] matrix = new int[][]{{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        int total = 0;
        int[][] grid = new int[n][m];
        UnionFind unionFind = new UnionFind(n * m);
        for (Point operator : operators) {
            if (grid[operator.x][operator.y] == 1) {
                result.add(total);
                continue;
            }
            total++;
            grid[operator.x][operator.y] = 1;

            unionFind.setCount(total);
            for (int[] direct : matrix) {
                int x = operator.x + direct[0];
                int y = operator.y + direct[1];
                if (!checkBound(x, y, n, m)) {
                    continue;
                }
                if (grid[x][y] == 1) {
                    unionFind.connect(operator.x * m + operator.y, x * m + y);
                }
            }
            total = unionFind.query();
            result.add(total);
        }
        return result;
    }

    private boolean checkBound(int x, int y, int n, int m) {
        return x >= 0 && x < n && y >= 0 && y < m;
    }


}
