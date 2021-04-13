package org.learn.algorithm.leetcode;

import org.learn.algorithm.datastructure.ListNode;

import javax.xml.crypto.dsig.keyinfo.RetrievalMethod;
import java.util.*;

/**
 * 第三页
 *
 * @author luk
 * @date 2021/4/13
 */
public class ThreePage {


    /**
     * 202. Happy Number
     */
    public boolean isHappy(int n) {
        Set<Integer> set = new HashSet<>();
        while (n != 1) {
            int tmp = n;
            int result = 0;
            while (tmp != 0) {
                int remain = tmp % 10;
                result = result + remain * remain;
                tmp /= 10;
            }
            if (!set.add(result)) {
                return false;
            }
            n = result;
        }
        return true;
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
            return removeElements(head.next, val);
        }
        head.next = removeElements(head.next, val);
        return head;
    }


    /**
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
        int m = s.length();
        int n = t.length();
        if (m != n) {
            return false;
        }
        Map<Character, Integer> map1 = new HashMap<>();
        Map<Character, Integer> map2 = new HashMap<>();
        for (int i = 0; i < m; i++) {
            if (!Objects.equals(map1.put(s.charAt(i), i), map2.put(t.charAt(i), i))) {
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
        ListNode node = reverseList(head.next);
        head.next.next = head;
        head.next = null;
        return node;
    }


    /**
     * todo use OlogN
     * 209. Minimum Size Subarray Sum
     *
     * @param target
     * @param nums
     * @return
     */
    public int minSubArrayLen(int target, int[] nums) {
        int left = 0;
        int right = 0;
        int local = 0;
        int result = Integer.MAX_VALUE;
        while (right < nums.length) {
            local += nums[right++];
            while (left < right && local >= target) {
                result = Math.min(result, right - left);
                local -= nums[left++];
            }
        }
        return result == Integer.MAX_VALUE ? 0 : result;
    }


}
