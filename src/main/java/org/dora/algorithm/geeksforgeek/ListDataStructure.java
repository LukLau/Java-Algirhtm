package org.dora.algorithm.geeksforgeek;

import org.dora.algorithm.datastructe.ListNode;

/**
 * @author dora
 * @date 2019/11/6
 */
public class ListDataStructure {


    /**
     * 82. Remove Duplicates from Sorted List II
     *
     * @param head
     * @return
     */
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        if (head.val == head.next.val) {
            ListNode nextNode = head.next.next;
            while (nextNode != null && nextNode.val == head.val) {
                nextNode = nextNode.next;
            }
            return this.deleteDuplicates(nextNode);
        } else {
            head.next = this.deleteDuplicates(head.next);
            return head;
        }
    }
}
