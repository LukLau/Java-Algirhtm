package org.learn.algorithm.leetcode;

import org.learn.algorithm.datastructure.ListNode;

import java.util.*;

/**
 * @author luk
 * @date 2021/4/6
 */
public class FirstPage {

    /**
     * 1. Two Sum
     *
     * @param nums
     * @param target
     * @return
     */
    public int[] twoSum(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return new int[]{};
        }
        int[] result = new int[2];
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(target - nums[i])) {
                result[0] = map.get(target - nums[i]);
                result[1] = i;
                return result;
            }
            map.put(nums[i], i);
        }
        return result;
    }

    /**
     * 2. Add Two Numbers
     *
     * @param l1
     * @param l2
     * @return
     */
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        if (l1 == null && l2 == null) {
            return null;
        }
        int carry = 0;
        ListNode root = new ListNode(0);
        ListNode dummy = root;
        while (l1 != null || l2 != null || carry != 0) {
            int val = (l1 == null ? 0 : l1.val) + (l2 == null ? 0 : l2.val) + carry;

            dummy.next = new ListNode(val % 10);

            carry = val / 10;

            dummy = dummy.next;
            l1 = l1 == null ? null : l1.next;
            l2 = l2 == null ? null : l2.next;
        }
        return root.next;
    }


    /**
     * 17. Letter Combinations of a Phone Number
     *
     * @param digits
     * @return
     */
    public List<String> letterCombinations(String digits) {
        if (digits == null || digits.isEmpty()) {
            return new ArrayList<>();
        }
        String[] map = new String[]{"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};

        int len = digits.length();

        LinkedList<String> deque = new LinkedList<>();

        deque.offer("");
        for (int i = 0; i < len; i++) {
            int index = Character.getNumericValue(digits.charAt(i));

            String word = map[index];

            while (deque.peek().length() == i) {
                String poll = deque.poll();

                char[] words = word.toCharArray();

                for (char t : words) {
                    deque.offer(poll + t);
                }
            }
        }
        return deque;
    }


    /**
     * 8. String to Integer (atoi)
     *
     * @param str
     * @return
     */
    public int myAtoi(String str) {
        if (str == null) {
            return 0;
        }
        str = str.trim();
        if (str.isEmpty()) {
            return 0;
        }
        char[] words = str.toCharArray();
        int index = 0;
        int sign = 1;
        if (words[index] == '+' || words[index] == '-') {
            sign = words[index] == '+' ? 1 : -1;
            index++;
        }
        long result = 0;
        while (index < words.length && Character.isDigit(words[index])) {
            result = result * 10 + Character.getNumericValue(words[index++]);
            if (result > Integer.MAX_VALUE) {
                return sign == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
            }
        }
        return (int) (result * sign);
    }


    /**
     * 32. Longest Valid Parentheses
     *
     * @param s
     * @return
     */
    public int longestValidParentheses(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        }
        Stack<Integer> stack = new Stack<>();
        char[] words = s.toCharArray();
        int result = 0;
        int left = -1;
        for (int i = 0; i < words.length; i++) {
            if (words[i] == '(') {
                stack.push(i);
            } else {
                if (!stack.isEmpty() && words[stack.peek()] == '(') {
                    stack.pop();
                } else {
                    left = i;
                }

                if (stack.isEmpty()) {
                    result = Math.max(result, i - left);
                } else {
                    result = Math.max(result, i - stack.peek());
                }
            }
        }
        return result;
    }

    public int longestValidParenthesesII(String s) {
        if (s == null) {
            return 0;
        }
        Stack<Integer> stack = new Stack<>();
        char[] words = s.toCharArray();
        for (int i = 0; i < words.length; i++) {
            if (words[i] == '(' || stack.isEmpty()) {
                stack.push(i);
            } else {
                if (words[stack.peek()] == '(') {
                    stack.pop();
                } else {
                    stack.push(i);
                }

            }
        }
        if (stack.isEmpty()) {
            return words.length;
        }
        int result = 0;
        int rightEdge = s.length();
        while (!stack.isEmpty()) {
            Integer left = stack.pop();
            result = Math.max(result, rightEdge - left - 1);
            rightEdge = left;
        }
        result = Math.max(result, rightEdge);
        return result;
    }

}
