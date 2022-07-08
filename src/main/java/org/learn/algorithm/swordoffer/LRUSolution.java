package org.learn.algorithm.swordoffer;

import java.util.HashMap;
import java.util.Map;

public class LRUSolution {

    private Map<Integer, Node> map = new HashMap<>();
    private int capacity = 0;

    private final Node head;

    private final Node tail;

    public LRUSolution(int capacity) {
        // write code here
        head = new Node();
        tail = new Node();
        head.next = tail;
        tail.prev = head;

        this.capacity = capacity;
    }

    public int get(int key) {
        // write code here
        Node current = map.get(key);

        if (current == null) {
            return -1;
        }
        transferNode(current);
        return current.value;
    }

    public void set(int key, int value) {
        // write code here
        Node node = map.get(key);
        if (node != null) {
            node.value = value;
            transferNode(node);
            return;
        }
        Node newNode = new Node(key, value);
        if (map.size() == capacity) {
            Node tailNode = removeTailNode();
            map.remove(tailNode.key);
        }
        map.put(key, newNode);
    }

    private void transferNode(Node current) {
        removeNode(current);
        insertHead(current);
    }

    private void removeNode(Node current) {
        Node prev = current.prev;
        Node tail = current.next;
        prev.next = tail;
        tail.prev = prev;

        current.prev = null;
        current.next = null;
    }

    private Node removeTailNode() {
        Node last = tail.prev;
        last.prev.next = tail;
        tail.prev = last.prev;

        last.prev = null;
        last.next = null;
        return last;
    }

    private void insertHead(Node current) {
        Node second = head.next;

        head.next = current;
        current.prev = head;

        current.next = second;
        second.prev = current;
    }

}


class Node {
    public int value;

    public int key;

    public Node next;

    public Node prev;

    public Node(int value) {
        this.value = value;
    }

    public Node(int key, int value) {
        this.key = key;
        this.value = value;
    }

    public Node() {

    }
}
