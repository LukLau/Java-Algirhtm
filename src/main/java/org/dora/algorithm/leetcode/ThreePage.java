package org.dora.algorithm.leetcode;

import org.dora.algorithm.datastructe.ListNode;

import java.util.HashSet;
import java.util.Set;

/**
 * @author dora
 * @date 2019-05-02
 */
public class ThreePage {


    /**
     * 201. Bitwise AND of Numbers Range
     * todo 不懂 位运算
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
        if (n <= 0) {
            return false;
        }
        Set<Integer> used = new HashSet<>();
        while (n != 0) {
            int tmp = n;
            int result = 0;
            while (tmp != 0) {
                int value = tmp % 10;
                result += value * value;

                tmp /= 10;
            }

            if (result == 1) {
                return true;
            }

            if (used.contains(result)) {
                return false;
            }
            n = result;
            used.add(n);
        }
        return false;
    }


    /**
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
     * 204. Count Primes 计算素数个数
     * todo 巧妙设计
     *
     * @param n
     * @return
     */
    public int countPrimes(int n) {
        int count = 0;
        for (int i = 2; i < Math.sqrt(n); i++) {
            if (this.isPrime(i)) {
                count++;
            }
        }
        return count;
    }

    private boolean isPrime(int i) {
        for (int j = 2; j < i; j++) {
            if (i % j == 0) {
                return false;
            }
        }
        return true;
    }

    /**
     * 205. Isomorphic Strings
     * todo 哈希思想 注意遍历退出条件
     *
     * @param s
     * @param t
     * @return
     */
    public boolean isIsomorphic(String s, String t) {
        if (s == null || t == null) {
            return false;
        }
        int[] hash1 = new int[512];
        int[] hash2 = new int[512];
        for (int i = 0; i < s.length(); i++) {
            if (hash1[s.charAt(i)] != hash2[t.charAt(i)]) {
                return false;
            }
            hash1[s.charAt(i)] = i + 1;
            hash2[t.charAt(i)] = i + 1;
        }
        return false;
    }

    /**
     * 反转链表
     *
     * @param head
     * @return
     */
    public ListNode reverseList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode next = this.reverseList(head.next);
        head.next.next = head;
        head.next = null;
        return next;

    }
}
