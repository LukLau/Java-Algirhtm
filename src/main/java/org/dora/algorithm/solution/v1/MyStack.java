package org.dora.algorithm.solution.v1;

import java.util.Deque;
import java.util.LinkedList;

/**
 * @author dora
 * @date 2019-05-03
 */
public class MyStack {

    private Deque<Integer> deque;

    /**
     * Initialize your data structure here.
     */
    public MyStack() {
        deque = new LinkedList<>();
    }

    /**
     * Push element x onto stack.
     */
    public void push(int x) {
        deque.offer(x);
        for (int i = 0; i < deque.size() - 1; i++) {
            int item = deque.poll();

            deque.offer(item);
        }
    }


    /**
     * Removes the element on top of the stack and returns that element.
     */
    public int pop() {
        if (this.empty()) {
            return -1;
        }
        return deque.pollFirst();

    }

    /**
     * Get the top element.
     */
    public int top() {
        if (this.empty()) {
            return -1;
        }
        return deque.peek();
    }

    /**
     * Returns whether the stack is empty.
     */
    public boolean empty() {
        return deque.isEmpty();
    }
}
