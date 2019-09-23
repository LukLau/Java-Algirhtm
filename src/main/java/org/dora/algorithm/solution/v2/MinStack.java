package org.dora.algorithm.solution.v2;

import java.util.Stack;

/**
 * @author dora
 * @date 2019/9/24
 */
public class MinStack {

    private Stack<Integer> stack;
    private Stack<Integer> minStack;

    /**
     * initialize your data structure here.
     */
    public MinStack() {
        stack = new Stack<>();
        minStack = new Stack<>();
    }

    public void push(int x) {
        stack.push(x);
        if (!minStack.isEmpty()) {
            int top = minStack.peek();
            if (x <= top) {
                minStack.push(x);
            }
        } else {
            minStack.push(x);
        }
    }

    public void pop() {
        Integer pop = stack.pop();
        if (pop <= minStack.peek()) {
            minStack.pop();
        }
    }

    public int top() {
        return stack.peek();
    }

    public int getMin() {
        return minStack.peek();
    }

}
