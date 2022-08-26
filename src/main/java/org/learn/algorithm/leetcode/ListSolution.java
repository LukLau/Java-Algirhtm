package org.learn.algorithm.leetcode;

import org.learn.algorithm.datastructure.ListNode;
import org.learn.algorithm.datastructure.RandomListNode;
import org.slf4j.event.SubstituteLoggingEvent;

import java.util.Comparator;
import java.util.PriorityQueue;

/**
 * 链表系列问题
 *
 * @author luk
 * @date 2021/4/7
 */
public class ListSolution {
    public static void main(String[] args) {
        ListSolution solution = new ListSolution();
        ListNode root = new ListNode(1);
        root.next = new ListNode(2);
        solution.removeNthFromEnd(root, 2);
    }


    // 链表交换问题//

    /**
     * 19. Remove Nth Node From End of List
     *
     * @param head
     * @param n
     * @return
     */
    public ListNode removeNthFromEnd(ListNode head, int n) {
        if (head == null) {
            return null;
        }
        ListNode root = new ListNode(0);
        root.next = head;

        ListNode fast = root;
        for (int i = 0; i < n; i++) {
            fast = fast.next;
        }
        ListNode slow = root;
        while (fast.next != null) {
            slow = slow.next;
            fast = fast.next;
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
        }
        return null;
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
            ListNode fast = dummy.next.next;
            ListNode slow = dummy.next;
            slow.next = fast.next;
            fast.next = slow;
            dummy.next = fast;

            dummy = dummy.next.next;

        }
        return root.next;
    }

    private ListNode reverse(ListNode start, ListNode end) {
        ListNode prev = end;
        while (start != end) {
            ListNode tmp = start.next;
            start.next = prev;
            prev = start;
            start = tmp;
        }
        return prev;
    }


    /**
     * 61. Rotate List
     *
     * @param head
     * @param k
     * @return
     */
    public ListNode rotateRight(ListNode head, int k) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode fast = head;
        int count = 1;

        while (fast.next != null) {
            fast = fast.next;
            count++;
        }
        fast.next = head;
        k %= count;
        ListNode slow = head;
        for (int i = 0; i < count - k; i++) {
            slow = slow.next;
            fast = fast.next;
        }
        fast.next = null;
        return slow;
    }


    /**
     * 86. Partition List
     *
     * @param head
     * @param x
     * @return
     */
    public ListNode partition(ListNode head, int x) {
        if (head == null || head.next == null) {
            return head;
        }

        ListNode slowNode = new ListNode(0);

        ListNode fastNode = new ListNode(0);

        ListNode slow = slowNode;
        ListNode fast = fastNode;
        while (head != null) {
            if (head.val < x) {
                slow.next = head;

                slow = slow.next;
            } else {
                fast.next = head;
                fast = fast.next;
            }
            head = head.next;
        }
        fast.next = null;
        slow.next = fastNode.next;
        return slowNode.next;
    }


    /**
     * 92. Reverse Linked List II
     *
     * @param head
     * @param left
     * @param right
     * @return
     */
    public ListNode reverseBetween(ListNode head, int left, int right) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode root = new ListNode(0);

        root.next = head;

        ListNode fast = root;
        ListNode slow = root;
        for (int i = 0; i < left - 1; i++) {
            slow = slow.next;
        }
        for (int i = 0; i < right; i++) {
            fast = fast.next;
        }
        ListNode endNode = fast.next;
        ListNode startNode = slow.next;

        slow.next = reverseNode(startNode, endNode);

        startNode.next = endNode;

        return root.next;
    }


    private ListNode reverseNode(ListNode head, ListNode tail) {
        ListNode end = tail;
        while (head != end) {
            ListNode tmp = head.next;
            head.next = tail;
            tail = head;
            head = tmp;
        }
        return tail;
    }

    private ListNode reverse(ListNode root) {
        ListNode prev = null;
        while (root != null) {
            ListNode tmp = root.next;
            root.next = prev;
            prev = root;
            root = tmp;
        }
        return prev;
    }


    /**
     * 138. Copy List with Random Pointer
     *
     * @param head
     * @return
     */
    public RandomListNode copyRandomList(RandomListNode head) {
        if (head == null) {
            return null;
        }
        RandomListNode current = head;
        while (current != null) {
            RandomListNode next = current.next;
            RandomListNode tmp = new RandomListNode(current.label);
            current.next = tmp;
            tmp.next = next;
            current = next;
        }
        current = head;
        while (current != null) {
            RandomListNode random = current.random;
            if (random != null) {
                current.next.random = random.next;
            }
            current = current.next.next;
        }
        RandomListNode copyOfHead = head.next;
        current = head;

        while (current.next != null) {
            RandomListNode tmp = current.next;
            current.next = tmp.next;
            current = tmp;
        }
        return copyOfHead;
    }


    /**
     * 141. Linked List Cycle
     *
     * @param head
     * @return
     */
    public boolean hasCycle(ListNode head) {
        if (head == null || head.next == null) {
            return false;
        }
        ListNode slow = head;
        ListNode fast = head;
        while (fast.next != null && fast.next.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast) {
                return true;
            }
        }
        return false;
    }


    /**
     * 142. Linked List Cycle II
     *
     * @param head
     * @return
     */
    public ListNode detectCycle(ListNode head) {
        if (head == null || head.next == null) {
            return null;
        }
        ListNode slow = head;
        ListNode fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast) {
                fast = head;
                while (fast != slow) {
                    fast = fast.next;
                    slow = slow.next;
                }
                return fast;
            }
        }
        return null;
    }


    /**
     * 143. Reorder List
     *
     * @param head
     */
    public void reorderList(ListNode head) {
        if (head == null || head.next == null) {
            return;
        }
        ListNode slow = head;
        ListNode fast = head;
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next;
            slow = slow.next;
        }
        ListNode second = slow.next;

        slow.next = null;

        ListNode reverse = reverse(second);

        ListNode first = head;

        while (first != null && reverse != null) {
            ListNode next = first.next;
            ListNode reverseNext = reverse.next;
            first.next = reverse;
            reverse.next = next;
            first = next;
            reverse = reverseNext;
        }

    }

}
