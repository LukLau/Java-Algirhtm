package org.learn.algorithm.leetcode;

import java.util.HashMap;
import java.util.Map;

/**
 * @author luk
 * @date 2021/4/13
 */
public class TwoSumIII {

    private Map<Integer, Integer> map = new HashMap<>();

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
        // write your code here
        for (Map.Entry<Integer, Integer> item : map.entrySet()) {
            Integer key = item.getKey();
            int diff = value - key;
            if (diff == key) {
                return item.getValue() >= 2;
            } else if (map.containsKey(diff)) {
                return true;
            }
        }
        return false;
    }

    public static void main(String[] args) {
        TwoSumIII solution = new TwoSumIII();
        solution.add(2);
        solution.add(3);
        solution.add(3);
        solution.find(6);
    }

}
