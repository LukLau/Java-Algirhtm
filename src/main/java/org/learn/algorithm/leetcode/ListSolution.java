package org.learn.algorithm.leetcode;

import org.learn.algorithm.datastructure.ListNode;
import org.learn.algorithm.datastructure.RandomListNode;

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
        ListNode current = head;
        int count = 1;
        while (current.next != null) {
            count++;
            current = current.next;
        }
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode fast = dummy;
        for (int i = 0; i < count - n; i++) {
            fast = fast.next;
        }
        fast.next = fast.next.next;
        return dummy.next;
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
        PriorityQueue<ListNode> priorityQueue = new PriorityQueue<>(Comparator.comparingInt(o -> o.val));
        for (ListNode node : lists) {
            if (node != null) {
                priorityQueue.offer(node);
            }
        }
        ListNode root = new ListNode(0);
        ListNode dummy = root;
        while (!priorityQueue.isEmpty()) {
            ListNode poll = priorityQueue.poll();
            dummy.next = poll;
            if (poll.next != null) {
                priorityQueue.offer(poll.next);
            }
            dummy = dummy.next;
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
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode root = dummy;
        while (root.next != null && root.next.next != null) {
            ListNode slow = root.next;
            ListNode fast = root.next.next;
            root.next = fast;
            ListNode next = fast.next;
            fast.next = slow;
            slow.next = next;
            root = root.next.next;
        }
        return dummy.next;
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
        ListNode current = head;
        int count = 1;
        while (current.next != null) {
            current = current.next;
            count++;
        }
        current.next = head;
        k %= count;
        ListNode fast = head;
        if (k != 0) {
            for (int i = 0; i < count - k; i++) {
                fast = fast.next;
                current = current.next;
            }
        }
        current.next = null;
        return fast;
    }


    /**
     * 86. Partition List
     *
     * @param head
     * @param x
     * @return
     */
    public ListNode partition(ListNode head, int x) {
        if (head == null) {
            return null;
        }
        ListNode small = new ListNode(0);
        ListNode big = new ListNode(0);

        ListNode d1 = small;
        ListNode d2 = big;
        while (head != null) {
            if (head.val < x) {
                d1.next = head;
                d1 = d1.next;
            } else {
                d2.next = head;
                d2 = d2.next;
            }
            head = head.next;
        }
        d1.next = big.next;
        d2.next = null;
        return small.next;
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
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode slow = dummy;
        ListNode fast = dummy;
        for (int i = 0; i < left - 1; i++) {
            slow = slow.next;
        }
        for (int i = 0; i < right; i++) {
            fast = fast.next;
        }
        ListNode node = slow.next;
        ListNode end = fast.next;
        fast.next = null;
        slow.next = null;
        slow.next = reverse(node);
        node.next = end;
        return dummy.next;
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
