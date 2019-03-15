package org.dora.algorithm.solution;

import java.util.HashMap;
import java.util.Map;

/**
 * @author liulu
 * @date 2019-03-15
 */
public class TwoSum {
    private HashMap<Integer, Integer> table = new HashMap<>();

    /**
     * @param number: An integer
     * @return: nothing
     */
    public void add(int number) {
        // write your code here
        int count = table.getOrDefault(number, 0);
        count = count + 1;
        table.put(number, count);

    }


    /**
     * @param value: An integer
     * @return: Find if there exists any pair of numbers which sum is equal to the value.
     */
    public boolean find(int value) {
        // write your code here
        for (Map.Entry<Integer, Integer> entry : table.entrySet()) {
            int num = entry.getKey();
            int y = value - num;
            if (y == num) {
                if (entry.getValue() >= 2) {
                    return true;
                }
            } else if (table.containsKey(y)) {
                return true;
            }

        }
        return false;
    }
}
