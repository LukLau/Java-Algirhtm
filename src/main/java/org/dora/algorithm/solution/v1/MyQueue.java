package org.dora.algorithm.solution.v1;

import java.util.Stack;

/**
 * @author dora
 * @date 2019-05-04
 */
public class MyQueue {
    Stack<Integer> s1;
    Stack<Integer> s2;

    /**
     * Initialize your data structure here.
     */
    public MyQueue() {
        s1 = new Stack<>();
        s2 = new Stack<>();

    }

    /**
     * Push element x to the back of queue.
     */
    public void push(int x) {
        s1.push(x);
    }

    /**
     * Removes the element from in front of queue and returns that element.
     */
    public int pop() {

        if (this.empty()) {
            return -1;
        }
        if (!s2.isEmpty()) {
            return s2.pop();
        }
        while (!s1.isEmpty()) {
            int top = s1.pop();
            s2.push(top);
        }
        return s2.pop();
    }

    /**
     * Get the front element.
     */
    public int peek() {
        if (this.empty()) {
            return -1;
        }
        if (!s2.isEmpty()) {
            return s2.peek();
        }
        while (!s1.isEmpty()) {
            int top = s1.pop();
            s2.push(top);
        }
        return s2.peek();
    }

    /**
     * Returns whether the queue is empty.
     */
    public boolean empty() {
        return s1.empty() && s2.isEmpty();
    }
}
