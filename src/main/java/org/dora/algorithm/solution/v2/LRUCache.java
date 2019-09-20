package org.dora.algorithm.solution.v2;

import java.util.HashMap;

/**
 * @author dora
 * @date 2019/9/21
 */
public class LRUCache {

    private int capacity;

    private Node head;

    private Node tail;

    private HashMap<Integer, Node> map;

    public LRUCache(int capacity) {
        this.capacity = capacity;

        head = new Node();

        tail = new Node();

        head.next = tail;

        tail.prev = head;

        map = new HashMap<>();

    }

    public int get(int key) {
        Node node = map.get(key);
        if (node == null) {
            return -1;
        }
        this.moveNode(node);
        return node.value;
    }

    private void removeNode(Node node) {
        Node prev = node.prev;
        Node next = node.next;

        prev.next = next;
        next.prev = prev;

        node.prev = null;
        node.next = null;
    }

    private void addFirstNode(Node node) {
        Node secondNode = head.next;

        secondNode.prev = node;
        node.next = secondNode;

        node.prev = head;
        head.next = node;
    }

    private void moveNode(Node node) {
        this.removeNode(node);
        this.addFirstNode(node);
    }

    public void put(int key, int value) {
        Node node = new Node(key, value);
        if (map.containsKey(key)) {
            node.value = value;
            map.put(key, node);
            this.moveNode(node);
        } else {

            map.put(key, node);

            this.addFirstNode(node);

            if (map.size() > capacity) {
                Node popNode = this.popTailNode();

                map.remove(popNode.key);
            }
        }
    }

    private Node popTailNode() {
        Node node = tail.prev;
        this.removeNode(node);
        return node;
    }

    class Node {
        private int key;
        private int value;

        private Node next;
        private Node prev;

        public Node() {

        }

        public Node(int key, int value) {
            this.key = key;
            this.value = value;
        }
    }
}



