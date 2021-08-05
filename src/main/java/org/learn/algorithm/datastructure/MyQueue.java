package org.learn.algorithm.datastructure;

import java.util.Stack;

/**
 * 使用两个
 *
 * @author luk
 * @date 2021/4/16
 */
public class MyQueue {

    private final Stack<Integer> push;

    private final Stack<Integer> pop;

    /**
     * Initialize your data structure here.
     */
    public MyQueue() {
        push = new Stack<>();
        pop = new Stack<>();
    }

    /**
     * Push element x to the back of queue.
     */
    public void push(int x) {
        push.push(x);
    }

    /**
     * Removes the element from in front of queue and returns that element.
     */
    public int pop() {
        if (!pop.isEmpty()) {
            return pop.pop();
        }
        while (!push.isEmpty()) {
            pop.push(push.pop());
        }
        return pop.pop();
    }

    /**
     * Get the front element.
     */
    public int peek() {
        if (!pop.isEmpty()) {
            return pop.peek();
        }
        while (!push.isEmpty()) {
            pop.push(push.pop());
        }
        return pop.peek();
    }

    /**
     * Returns whether the queue is empty.
     */
    public boolean empty() {
        return pop.isEmpty() && !push.isEmpty();
    }

}
