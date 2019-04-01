package org.dora.algorithm.solution;

import java.util.HashMap;


/**
 * @author liulu
 * @date 2019-03-14
 */
public class LRUCache {

    private int count;
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
        moveNode(node);
        return node.value;
    }

    private void moveNode(Node node) {
        removeNode(node);
        addNode(node);
    }

    private void addNode(Node node) {
        Node tmp = head.next;

        node.prev = head;
        head.next = node;

        tmp.prev = node;
        node.next = tmp;
    }

    private void removeNode(Node node) {
        Node tmp = node.prev;
        Node next = node.next;
        tmp.next = next;
        next.prev = tmp;

        node.prev = null;
        node.next = null;
    }

    public void put(int key, int value) {
        Node node = map.get(key);
        if (node != null) {
            node.value = value;
            moveNode(node);
        } else {
            node = new Node(key, value);
            map.put(key, node);
            addNode(node);
            count++;
            if (count > capacity) {
                Node tmp = getTailNode();
                map.remove(tmp.key);
                count--;
            }
        }
    }

    private Node getTailNode() {
        Node node = tail.prev;
        removeNode(node);
        return node;
    }

    class Node {
        Node prev;
        Node next;
        private int key;
        private int value;

        private Node() {
        }

        private Node(int key, int value) {
            this.key = key;
            this.value = value;
        }
    }
}
