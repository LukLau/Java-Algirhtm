package org.dora.algorithm.solution;

import java.util.HashMap;

/**
 * @author liulu
 * @date 2019-03-14
 */
public class LRUCache {
    private int capacity;
    private int count;
    private Node head;
    private Node tail;
    private HashMap<Integer, Node> map = new HashMap<>();

    public LRUCache(int capacity) {
        this.capacity = capacity;

        head = new Node();

        tail = new Node();


        head.next = tail;

        tail.prev = head;
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
        addFirst(node);
    }


    private void addFirst(Node node) {
        node.prev = head;
        node.next = head.next;

        head.next.prev = node;
        head.next = node;
    }

    private void removeNode(Node node) {
        Node prev = node.prev;

        Node post = node.next;

        prev.next = post;

        post.prev = prev;

        node.prev = null;
        node.next = null;

    }

    public void put(int key, int value) {
        Node node = map.get(key);
        if (node != null) {
            node.value = value;
            moveNode(node);
            return;
        }
        node = new Node(key, value);
        map.put(key, node);
        addFirst(node);
        count++;
        if (count > capacity) {
            Node tmp = popTail();
            map.remove(tmp.key);
            count--;
        }
    }

    private Node popTail() {
        Node node = tail.prev;

        removeNode(node);

        return node;
    }

    class Node {
        int key;
        int value;
        Node prev;
        Node next;

        public Node(int key, int value) {
            this.key = key;
            this.value = value;
        }

        public Node() {
        }
    }


}
