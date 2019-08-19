package org.dora.algorithm.solution.v1;

import java.util.HashMap;

/**
 * @author liulu
 * @date 2019-03-14
 */
public class LRUCache {


    private volatile int capacity;


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

        this.moveNode(node);

        return node.value;
    }

    private void moveNode(Node node) {
        this.removeNode(node);
        this.addNode(node);
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

    private Node popNode() {
        Node prev = tail.prev;

        this.removeNode(prev);

        return prev;
    }

    public void put(int key, int value) {
        Node node = map.get(key);
        if (node != null) {
            node.value = value;
            this.moveNode(node);
            return;
        } else {

            node = new Node(key, value);


            if (map.size() == capacity) {

                Node tail = this.popNode();

                map.remove(tail.key);
            }

            map.put(key, node);

            this.addNode(node);
        }
    }


    /**
     * 内部数据结构
     */
    class Node {
        int key;
        int value;

        Node prev;
        Node next;

        public Node() {
        }

        public Node(int key, int value) {
            this.key = key;
            this.value = value;
        }
    }


}
