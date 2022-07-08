package org.learn.algorithm.leetcode;

import java.util.HashMap;
import java.util.Map;

/**
 * LRU last recently unused
 *
 * @author luk
 * @date 2021/4/12
 */
public class LRUCache {
    private final Map<Integer, Node> map;
    private final int capacity;
    private final Node head;
    private final Node tail;


    public LRUCache(int capacity) {
        map = new HashMap<>();
        head = new Node();
        tail = new Node();
        head.next = tail;
        tail.prev = head;

        this.capacity = capacity;
    }

    public int get(int key) {

        Node node = map.get(key);
        if (node == null) {
            return -1;
        }
        addToFirst(node);

        return node.val;
    }

    public void put(int key, int value) {
        Node current = map.get(key);

        if (current != null) {
            current.val = value;
            addToFirst(current);
            return;
        }
        Node node = new Node(key, value);

        if (map.size() == this.capacity) {
            Node tail = getTail();
            map.remove(tail.key);
        }
        map.put(key, node);

        addNode(node);

    }


    private void addToFirst(Node node) {
        removeNode(node);
        addNode(node);
    }

    private void addNode(Node node) {
        Node previous = head.next;
        head.next = node;
        node.prev = head;

        previous.prev = node;
        node.next = previous;

    }

    private Node getTail() {
        Node prev = tail.prev;
        prev.prev.next = tail;
        tail.prev = prev.prev;

        prev.prev = null;
        prev.next = null;
        return prev;
    }

    private void removeNode(Node node) {
        Node prev = node.prev;

        prev.next = node.next;

        node.next.prev = prev;

        node.prev = null;
        node.next = null;
    }


    static class Node {
        private int val;
        private int key;
        private Node next;
        private Node prev;

        public Node(int key, int val) {
            this.val = val;
            this.key = key;
        }

        public Node() {
        }
    }

}
