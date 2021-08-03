package org.learn.algorithm.datastructure;

import java.util.LinkedList;

/**
 * @author luk
 * @date 2021/8/3
 */
public class MyStack {

    private final LinkedList<Integer> linkedList;

    /**
     * Initialize your data structure here.
     */
    public MyStack() {
        linkedList = new LinkedList<>();
    }

    /**
     * Push element x onto stack.
     */
    public void push(int x) {
        linkedList.offer(x);
    }

    /**
     * Removes the element on top of the stack and returns that element.
     */
    public int pop() {
        int size = linkedList.size();
        for (int i = 0; i < size - 1; i++) {
            Integer poll = linkedList.poll();
            linkedList.offer(poll);
        }
        return linkedList.poll();
    }

    /**
     * Get the top element.
     */
    public int top() {
        return linkedList.get(linkedList.size() - 1);
    }

    /**
     * Returns whether the stack is empty.
     */
    public boolean empty() {
        return linkedList.isEmpty();
    }
}
