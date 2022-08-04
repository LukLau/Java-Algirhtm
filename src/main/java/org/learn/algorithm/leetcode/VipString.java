package org.learn.algorithm.leetcode;

import org.springframework.context.event.EventListenerMethodProcessor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class VipString {

    /**
     * @param s: the given string
     * @return: if a permutation of the string could form a palindrome
     */
    public boolean canPermutePalindrome(String s) {
        // write your code here
        if (s == null || s.isEmpty()) {
            return false;
        }
        char[] words = s.toCharArray();
        Map<Character, Integer> map = new HashMap<>();
        for (char word : words) {
            int count = map.getOrDefault(word, 0);
            count++;
            map.put(word, count);
        }
        boolean odd = false;
        for (Map.Entry<Character, Integer> item : map.entrySet()) {
            Integer count = item.getValue();
            if (count % 2 != 0) {
                if (odd) {
                    return false;
                }
                odd = true;
            }
        }
        return true;
    }

    /**
     * @param s: the given string
     * @return: all the palindromic permutations (without duplicates) of it
     * we will sort your return value in output
     */
    public List<String> generatePalindromes(String s) {
        // write your code here
        if (s == null || s.isEmpty()) {
            return new ArrayList<>();
        }
        char[] words = s.toCharArray();
        Map<Character, Integer> map = new HashMap<>();
        for (char word : words) {
            int count = map.getOrDefault(word, 0);
            count++;
            map.put(word, count);
        }
        String odd = "";
        List<String> evens = new ArrayList<>();
        for (Map.Entry<Character, Integer> item : map.entrySet()) {
            Character key = item.getKey();
            int count = item.getValue();
            if (count % 2 != 0) {
                if (!odd.equals(String.valueOf(key))) {
                    return new ArrayList<>();
                }
                odd = String.valueOf(key);
            }
            StringBuilder builder = new StringBuilder();
            for (int j = 0; j < count / 2; j++) {
                builder.append(key);
            }
            evens.add(builder.toString());
        }
        return null;
    }

    private void generateWords(List<String> generateWords, List<String> evens, String s, boolean[] used) {

        for (int i = 0; i < evens.size(); i++) {

        }

    }


}
