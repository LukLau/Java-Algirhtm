package org.dora.algorithm.datastructe;

import java.util.List;

/**
 * @author liulu
 * @date 2019-03-12
 */
public class Node {

    public int val;
    public Node next;
    public Node random;


    public Node left;
    public Node right;

    public List<Node> neighbors;

    public Node(int val, Node next, Node random) {
        this.val = val;
        this.next = next;
        this.random = random;
    }

    public Node(int val) {
        this.val = val;
    }
}
