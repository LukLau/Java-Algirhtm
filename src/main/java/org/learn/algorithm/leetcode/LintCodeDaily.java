package org.learn.algorithm.leetcode;


import javax.swing.text.html.HTMLEditorKit;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

public class LintCodeDaily {

    public static void main(String[] args) {

        LintCodeDaily lintCodeDaily = new LintCodeDaily();

        int[] ints = lintCodeDaily.calculateNumber(7);
        System.out.println(Arrays.toString(ints));
    }

    /**
     * https://www.lintcode.com/problem/1665/
     *
     * @param num: the num
     * @return: the array subject to the description
     */
    public int[] calculateNumber(int num) {
        // Write your code here.
        int count = 0;
        List<Integer> position = new ArrayList<>();
        int index = 0;
        for (int i = 0; i < 32; i++) {
            if ((num & (1 << i)) != 0) {
                count++;
                position.add(index++);
            }
        }
        int[] result = new int[position.size() + 1];

        result[0] = count;
        for (int i = 1; i <= result.length; i++) {
            result[i] = position.get(i - 1);
        }
        return result;

    }

}
