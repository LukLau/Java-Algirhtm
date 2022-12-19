package org.learn.algorithm.leetcode;

import org.learn.algorithm.datastructure.ListNode;
import org.learn.algorithm.datastructure.Node;

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
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode fast = dummy;
        ListNode slow = fast;
        for (int i = 0; i < n; i++) {
            fast = fast.next;
        }
        while (fast.next != null) {
            slow = slow.next;
            fast = fast.next;
        }
        slow.next = slow.next.next;
        return dummy.next;
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
        for (ListNode list : lists) {
            if (list != null) {
                priorityQueue.offer(list);
            }
        }
        ListNode root = null;
        ListNode dummy = null;
        while (!priorityQueue.isEmpty()) {
            ListNode poll = priorityQueue.poll();
            if (root == null) {
                root = poll;
                dummy = poll;
            } else {
                dummy.next = poll;
                dummy = dummy.next;
            }
            if (poll.next != null) {
                priorityQueue.offer(poll.next);
            }
        }
        return root;
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
        if (head == null) {
            return null;
        }
        ListNode fast = head;
        int count = 1;
        while (fast.next != null) {
            fast = fast.next;
            count++;
        }
        fast.next = head;
        ListNode slow = head;
        k %= count;
        if (k != 0) {
            for (int i = 0; i < count - k; i++) {
                fast = fast.next;
                slow = slow.next;
            }
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
        ListNode small = new ListNode(0);
        ListNode big = new ListNode(0);

        ListNode s = small;
        ListNode b = big;
        while (head != null) {
            if (head.val < x) {
                s.next = head;
                s = s.next;
            } else {
                b.next = head;
                b = b.next;
            }
            head = head.next;
        }
        s.next = big.next;
        b.next = null;
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
        if (head == null) {
            return null;
        }
        ListNode root = new ListNode(0);
        root.next = head;
        ListNode slow = root;
        for (int i = 0; i < left - 1; i++) {
            slow = slow.next;
        }
        ListNode fast = root;
        for (int i = 0; i < right; i++) {
            fast = fast.next;
        }
        ListNode start = slow.next;
        ListNode end = fast.next;

        slow.next = reverseNode(start, end);

        start.next = end;

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
    public Node copyRandomList(Node head) {
        if (head == null) {
            return null;
        }
        Node current = head;
        while (current != null) {
            Node randomListNode = new Node(current.val);

            randomListNode.next = current.next;

            current.next = randomListNode;

            current = randomListNode.next;
        }
        current = head;
        while (current != null) {
            if (current.random != null) {
                current.next.random = current.random.next;
            }
            current = current.next.next;
        }
        current = head;
        Node copy = head.next;
        while (current.next != null) {
            Node tmp = current.next;
            current.next = tmp.next;
            current = tmp;
        }
        return copy;
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
        ListNode fast = head;
        ListNode slow = head;
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        ListNode next = slow.next;

        slow.next = null;

        ListNode reverse = reverse(next);

        fast = head;
        while (fast != null && reverse != null) {
            ListNode tmp = reverse.next;

            reverse.next = fast.next;

            fast.next = reverse;

            reverse = tmp;

            fast = fast.next.next;
        }
    }

}
