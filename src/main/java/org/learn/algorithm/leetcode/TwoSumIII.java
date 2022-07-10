package org.learn.algorithm.leetcode;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

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
        int count = map.getOrDefault(number, 0);
        map.put(number, count + 1);
    }

    /**
     * @param value: An integer
     * @return: Find if there exists any pair of numbers which sum is equal to the value.
     */
    public boolean find(int value) {
        // write your code here
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            int number = entry.getKey();

            int remain = value - number;

            if (remain == number) {
                if (entry.getValue() > 1) {
                    return true;
                }
                return false;
            }
            if (map.containsKey(remain)) {
                return true;
            }
        }
        return false;
    }


}
