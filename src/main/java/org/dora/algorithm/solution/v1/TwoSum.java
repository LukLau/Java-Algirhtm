package org.dora.algorithm.solution.v1;

import java.util.HashMap;
import java.util.Map;

/**
 * @author liulu
 * @date 2019-03-15
 */
public class TwoSum {

    private HashMap<Integer, Integer> map = new HashMap<>();

    /**
     * @param number: An integer
     * @return: nothing
     */
    public void add(int number) {
        // write your code here
        int count = map.getOrDefault(number, 0);
        map.put(number, ++count);
    }

    /**
     * @param value: An integer
     * @return: Find if there exists any pair of numbers which sum is equal to the value.
     */
    public boolean find(int value) {
        // write your code here
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {

            int key = entry.getKey();

            int remain = value - key;

            if (remain == key) {
                return entry.getValue() >= 2;
            } else {
                if (map.containsKey(remain)) {
                    return true;
                }
            }
        }
        return false;
    }
}
