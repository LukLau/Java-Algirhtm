package org.learn.algorithm.leetcode;

import org.learn.algorithm.datastructure.ListNode;

import java.util.Comparator;
import java.util.PriorityQueue;

/**
 * 链表系列问题
 *
 * @author luk
 * @date 2021/4/7
 */
public class ListSolution {


    // 链表交换问题//

    /**
     * 19. Remove Nth Node From End of List
     *
     * @param head
     * @param n
     * @return
     */
    public ListNode removeNthFromEnd(ListNode head, int n) {
        if (head == null || n <= 0) {
            return null;
        }
        ListNode current = head;
        int count = 1;
        while (current.next != null) {
            current = current.next;
            count++;
        }
        ListNode root = new ListNode(0);
        root.next = head;
        ListNode dummy = root;
        for (int i = 0; i < count - n; i++) {
            dummy = dummy.next;
        }
        dummy.next = dummy.next.next;

        return root.next;
    }


    public ListNode removeNthFromEndII(ListNode head, int n) {
        if (n <= 0 || head == null) {
            return null;
        }
        ListNode root = new ListNode(0);

        root.next = head;

        ListNode fast = root;

        for (int i = 0; i < n; i++) {
            if (fast == null) {
                return null;
            }
            fast = fast.next;
        }
        ListNode slow = root;
        while (fast.next != null) {
            fast = fast.next;
            slow = slow.next;
        }
        slow.next = slow.next.next;

        return root.next;
    }


    /**
     * 23. Merge k Sorted Lists
     *
     * @param lists
     * @return
     */
    public ListNode mergeKLists(ListNode[] lists) {
        if (lists == null || lists.length == 0) {
            return null;
        }
        PriorityQueue<ListNode> queue = new PriorityQueue<>(lists.length, Comparator.comparingInt(o -> o.val));
        for (ListNode node : lists) {
            if (node != null) {
                queue.offer(node);
            }
        }
        ListNode root = new ListNode(0);
        ListNode dummy = root;
        while (!queue.isEmpty()) {
            ListNode node = queue.poll();
            dummy.next = node;
            dummy = dummy.next;

            if (node.next != null) {
                queue.offer(node.next);
            }

        }
        return root.next;
    }


    /**
     * 24. Swap Nodes in Pairs
     *
     * @param head
     * @return
     */
    public ListNode swapPairs(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode root = new ListNode(0);
        root.next = head;
        ListNode dummy = root;
        while (dummy.next != null && dummy.next.next != null) {

            ListNode slow = dummy.next;

            ListNode fast = dummy.next.next;

            dummy.next = fast;

            slow.next = fast.next;

            fast.next = slow;

            dummy = slow;

        }
        return root.next;
    }
}
