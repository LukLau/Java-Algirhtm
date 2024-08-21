package org.dora.algorithm.swordoffer;


import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;

/**
 * mail lu.liu2@cariad-technology.cn
 * date 2024年04月05日
 *
 * @author lu.liu2
 */
public class LRU {

    private final Map<Integer, Node> map;

    private Node head = null;

    private Node tail = null;


    public LRU(int capacity) {
        // write code here
        map = new HashMap<>(capacity);

        head = new Node(-1);

        tail = new Node(-1);

        head.next = tail;
        tail.prev = head;

    }

    public int get(int key) {
        // write code here
        Node node = map.get(key);
        moveToFirst(node);
        return node.val;
    }


    private void moveToFirst(Node node) {
        Node prev = node.prev;
        Node tail = node.next;

        prev.next = tail;
        tail.prev = prev;


        Node tmp = head.next;

        head.next = node;
        node.prev = head;


        node.next = tmp;
        tmp.prev = node;
    }

    public void set(int key, int value) {
        // write code here

        Node node = new Node(value);

    }

    class Node {
        private int val;

        private Node prev;

        private Node next;

        public Node(int val) {
            this.val = val;
        }
    }

}
