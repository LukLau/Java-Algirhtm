package org.dora.algorithm.solution.v2;

import org.dora.algorithm.datastructe.ListNode;

import java.util.HashSet;
import java.util.Set;

/**
 * @author dora
 * @date 2019/9/30
 */
public class ThreeHundred {


    /**
     * todo 难题
     * 201. Bitwise AND of Numbers Range
     *
     * @param m
     * @param n
     * @return
     */
    public int rangeBitwiseAnd(int m, int n) {
        return 0;
    }


    /**
     * 202. Happy Number
     *
     * @param n
     * @return
     */
    public boolean isHappy(int n) {
        if (n < 0) {
            return false;
        }

        Set<Integer> set = new HashSet<>();

        while (n != 0) {
            int result = n;

            int value = 0;

            while (result != 0) {
                int tmp = result % 10;

                value += tmp * tmp;

                result /= 10;
            }
            if (set.contains(value)) {
                return false;
            }

            if (value == 1) {
                return true;
            }
            set.add(value);

            n = value;

        }
        return false;
    }


    /**
     * tdo 搞不懂答案
     * 203. Remove Linked List Elements
     *
     * @param head
     * @param val
     * @return
     */
    public ListNode removeElements(ListNode head, int val) {
        if (head == null) {
            return null;
        }
        if (head.val == val) {
            return this.removeElements(head.next, val);
        } else {
            head.next = this.removeElements(head.next, val);
            return head;
        }
    }


    /**
     * todo 求解质数
     * 204. Count Primes
     *
     * @param n
     * @return
     */
    public int countPrimes(int n) {
        if (n <= 0) {
            return 0;
        }
        int count = 0;

        for (int i = 2; i <= n; i++) {

            for (int j = 2; j * j <= n; j++) {

                if (i % 2 == 0) {
                    count++;
                }
            }
        }
        return count;
    }

    /**
     * todo 不懂
     * 205. Isomorphic Strings
     *
     * @param s
     * @param t
     * @return
     */
    public boolean isIsomorphic(String s, String t) {
        if (s == null || t == null) {
            return false;
        }
        if (s.length() != t.length()) {
            return false;
        }
        int[] hash1 = new int[256];
        int[] hash2 = new int[256];
        for (int i = 0; i < s.length(); i++) {
            hash1[s.charAt(i) - 'a']++;
            hash2[t.charAt(i) - 'a']++;
        }
        for (int i = 0; i < hash1.length; i++) {
            if (hash1[i] != hash2[i]) {
                return false;
            }
        }
        return true;
    }

    /**
     * 206. Reverse Linked List
     *
     * @param head
     * @return
     */
    public ListNode reverseList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode tmp = head.next;

        ListNode node = this.reverseList(tmp);

        tmp.next = head;

        head.next = null;

        return node;

    }

    /**
     * todo 不懂
     * 207. Course Schedule
     *
     * @param numCourses
     * @param prerequisites
     * @return
     */
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        return false;
    }

    /**
     * 209. Minimum Size Subarray Sum
     *
     * @param s
     * @param nums
     * @return
     */
    public int minSubArrayLen(int s, int[] nums) {
        if (nums == null || nums.length == 0) {
            return -1;
        }
        int result = 0;
        int begin = 0;
        int end = 0;
        while (end < nums.length) {

        }
        return result;
    }
}
