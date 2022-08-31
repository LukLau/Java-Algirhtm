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
            Integer entryKey = entry.getKey();
            int remain = value - entryKey;

            if (remain == entryKey) {
                return entry.getValue() > 1;
            } else if (map.containsKey(remain)) {
                return true;
            }
        }
        return false;
    }

    public static void main(String[] args) {
        TwoSumIII twoSum = new TwoSumIII();
        twoSum.add(3);
        twoSum.add(3);

        System.out.println(twoSum.find(6));

    }


}
