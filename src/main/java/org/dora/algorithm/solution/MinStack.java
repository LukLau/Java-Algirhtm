package org.dora.algorithm.solution;

import java.util.Stack;

/**
 * @author liulu
 * @date 2019-03-14
 */
public class MinStack {
    private Stack<Integer> minStack;
    private Stack<Integer> stack;

    /** initialize your data structure here. */
    public MinStack() {
        minStack = new Stack<>();
        stack = new Stack<>();
    }

    public void push(int x) {
        stack.push(x);
        if (minStack.isEmpty()) {
            minStack.push(x);
        } else {
            int top = minStack.peek();
            if (top >= x) {
                minStack.push(x);
            }
        }
    }

    public void pop() {
        int top = stack.pop();
        if (!minStack.isEmpty()) {
            int min = minStack.peek();
            if (top <= min) {
                minStack.pop();
            }
        }
    }

    public int top() {
        return stack.peek();
    }

    public int getMin() {
        return minStack.peek();
    }
}
