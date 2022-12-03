package org.learn.algorithm.leetcode;

import org.springframework.context.event.EventListenerMethodProcessor;

import java.util.*;

public class VipString {

    public static void main(String[] args) {
        VipString vipString = new VipString();
        vipString.generatePalindromes("aab");
    }


    /**
     * 161 One Edit Distance
     *
     * @param s: a string
     * @param t: a string
     * @return: true if they are both one edit distance apart or false
     */
    public boolean isOneEditDistance(String s, String t) {
        // write your code here
        if (s == null || t == null) {
            return false;
        }
        if (s.equals(t)) {
            return true;
        }
        return false;
    }


    /**
     * 266 Palindrome Permutation
     *
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
     * 267 Palindrome Permutation II
     *
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
        StringBuilder builder = new StringBuilder();
        for (Map.Entry<Character, Integer> item : map.entrySet()) {
            Character key = item.getKey();
            int count = item.getValue();
            if (count % 2 != 0) {
                if (!odd.isEmpty() && !odd.equals(String.valueOf(key))) {
                    return new ArrayList<>();
                }
                odd = String.valueOf(key);
            }
            for (int j = 0; j < count / 2; j++) {
                builder.append(key);
            }
        }


        List<String> params = new ArrayList<>();

        generateWords(params, 0, builder.toString().toCharArray());

        List<String> result = new ArrayList<>();

        for (String param : params) {
            String reverse = new StringBuilder(param).reverse().toString();
            result.add(param + odd + reverse);
        }
        return result;
    }

    private void generateWords(List<String> result, int start, char[] words) {
        if (start == words.length) {
            result.add(String.valueOf(words));
            return;
        }
        for (int i = start; i < words.length; i++) {
            if (i > start && words[i] == words[start]) {
                continue;
            }
            swap(words, i, start);
            generateWords(result, start + 1, words);
            swap(words, i, start);
        }
    }


    private void swap(char[] words, int i, int j) {
        char tmp = words[i];
        words[i] = words[j];
        words[j] = tmp;
    }


    /**
     * 273. Integer to English Words
     *
     * @param num
     * @return
     */
    public String numberToWords(int num) {
        return "";
    }


//    411. Minimum Unique Word Abbreviation


    /**
     * 411. Minimum Unique Word Abbreviation
     */
    public String[] wordsAbbreviation(String[] dict) {
        if (dict == null || dict.length == 0) {
            return dict;
        }
        // write your code here

        Map<String, PriorityQueue<String>> map = new HashMap<>();

        for (String word : dict) {
            String abbr = getAbbr(word);
            PriorityQueue<String> priorityQueue = map.getOrDefault(abbr, new PriorityQueue<>((o1, o2) -> o2.length() - o1.length()));

            if (priorityQueue.isEmpty()) {
                priorityQueue.offer(word);
            } else {
                boolean existUnique = true;
                for (String prefix : priorityQueue) {


                }
            }
        }
        return null;
    }

    private String getAbbr(String word) {

        int len = word.length();
        if (len <= 2) {
            return word;
        }
        String abbr = String.valueOf(word.charAt(0) + len + word.charAt(len - 1));
        if (abbr.length() == len) {
            return word;
        }
        return abbr;
    }


}
