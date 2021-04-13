package org.learn.algorithm.leetcode;

import org.learn.algorithm.datastructure.ListNode;

import java.util.HashSet;
import java.util.Set;

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
        return null;

    }

}
