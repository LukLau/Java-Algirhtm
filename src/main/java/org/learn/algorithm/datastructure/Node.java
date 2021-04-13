package org.learn.algorithm.datastructure;

/**
 * @author luk
 * @date 2021/4/11
 */
public class Node {

    public int val;
    public Node left;
    public Node right;
    public Node next;
    public Node random;

    public Node() {}

    public Node(int val) {
        this.val = val;
    }

    public Node(int val, Node left, Node right, Node next) {
        this.val = val;
        this.left = left;
        this.right = right;
        this.next = next;
    }
}
