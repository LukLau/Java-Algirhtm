package org.learn.algorithm.leetcode;

import org.thymeleaf.expression.Strings;

import java.util.*;

/**
 * leetcode the four page
 *
 * @author luk
 * @date 2021/4/26
 */
public class FourPage {

    public static void main(String[] args) {
        FourPage fourPage = new FourPage();
        String s = "cbacdcbc";
        System.out.println(fourPage.removeDuplicateLetters(s));
    }


    /**
     * 301. Remove Invalid Parentheses
     *
     * @param s
     * @return
     */
    public List<String> removeInvalidParentheses(String s) {
        if (s == null) {
            return new ArrayList<>();
        }
        List<String> result = new ArrayList<>();
        LinkedList<String> linkedList = new LinkedList<>();
        linkedList.offer(s);
        Set<String> seen = new HashSet<>();
        while (!linkedList.isEmpty()) {
            String poll = linkedList.poll();
            if (checkValid(poll) && !result.contains(poll)) {
                result.add(poll);
            }
            if (!result.isEmpty()) {
                continue;
            }
            int len = poll.length();
            for (int i = 0; i < len; i++) {
                char word = poll.charAt(i);
                if (word != '(' && word != ')') {
                    continue;
                }
                String tmp = poll.substring(0, i) + poll.substring(i + 1);
                if (!seen.contains(tmp)) {
                    linkedList.offer(tmp);
                    seen.add(tmp);
                }
            }
        }
        if (result.isEmpty()) {
            result.add("");
        }
        return result;
    }

    private boolean checkValid(String s) {
        int count = 0;
        char[] words = s.toCharArray();
        for (char word : words) {
            if (word != '(' && word != ')') {
                continue;
            }
            if (word == '(') {
                count++;
            }
            if (word == ')') {
                if (count == 0) {
                    return false;
                }
                count--;
            }
        }
        return count == 0;
    }


    public boolean isAdditiveNumber(String num) {
        if (num == null || num.isEmpty()) {
            return true;
        }
        int len = num.length();
        for (int i = 1; i < len; i++) {
            if (num.charAt(0) == '0' && i > 1) {
                return false;
            }
            for (int j = i + 1; len - j >= j - i && len - j >= i; j++) {
                if (num.charAt(j) == '0' && j - i > 2) {
                    break;
                }
                long num1 = Long.parseLong(num.substring(0, i));
                long num2 = Long.parseLong(num.substring(i, i + j));
                if (isAdditive(num.substring(i + j), num1, num2)) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean isAdditive(String substring, long num1, long num2) {
        if (substring.isEmpty()) {
            return true;
        }
        long sum = num1 + num2;
        String value = String.valueOf(sum);
        if (!substring.startsWith(value)) {
            return false;
        }
        return isAdditiveNumber(substring.substring(value.length()));
    }


    /**
     * 316. Remove Duplicate Letters
     *
     * @param s
     * @return
     */
    public String removeDuplicateLetters(String s) {
        if (s == null || s.isEmpty()) {
            return "";
        }
        return "";
    }
}
