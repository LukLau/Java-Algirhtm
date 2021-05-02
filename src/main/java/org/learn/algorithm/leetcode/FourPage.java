package org.learn.algorithm.leetcode;

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

        System.out.println(fourPage.removeInvalidParentheses("()())()"));
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
}
