package org.learn.algorithm.leetcode;

import java.util.HashMap;
import java.util.Map;

/**
 * @author luk
 * @date 2021/4/13
 */
public class TwoSumIII {

    private final Map<Integer, Integer> map = new HashMap<>();

    /**
     * @param number: An integer
     * @return: nothing
     */
    public void add(int number) {
        // write your code here
        Integer count = map.getOrDefault(number, 0);
        map.put(number, count + 1);
    }

    /**
     * @param value: An integer
     * @return: Find if there exists any pair of numbers which sum is equal to the value.
     */
    public boolean find(int value) {
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            Integer key = entry.getKey();
            int remain = value - key;

            if (remain == key) {
                Integer count = entry.getValue();
                return count > 1;
            } else if (map.containsKey(remain)) {
                return true;
            }
        }
        return false;
        // write your code here
    }

    public static void main(String[] args) {
        TwoSumIII solution = new TwoSumIII();
        solution.add(2);
        solution.add(3);
        solution.add(3);
        solution.find(6);
    }

}
