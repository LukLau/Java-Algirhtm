package org.learn.algorithm.datastructure;

/**
 * @author luk
 * @date 2021/5/12
 */
public class RandomListNode {
    public int label;

    private int value;

    public RandomListNode next = null;
    public RandomListNode random = null;

    public RandomListNode(int label) {
        this.label = label;
    }
}
