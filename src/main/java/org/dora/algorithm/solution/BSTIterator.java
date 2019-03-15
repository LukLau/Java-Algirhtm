package org.dora.algorithm.solution;

import org.dora.algorithm.datastructe.TreeNode;

import java.util.LinkedList;
import java.util.Stack;

/**
 * @author liulu
 * @date 2019-03-15
 */
public class BSTIterator {
    LinkedList<Integer> linkedList = new LinkedList<>();
    public BSTIterator(TreeNode root) {

        Stack<TreeNode> stack = new Stack<>();

        TreeNode p = root;

        while (!stack.isEmpty() || p != null) {
            while (p != null) {
                stack.push(p);
                p = p.left;
            }
            p = stack.pop();

            linkedList.addLast(p.val);

            p = p.right;
        }
    }

    /**
     * @return the next smallest number
     */
    public int next() {
        return linkedList.pollFirst();
    }

    /**
     * @return whether we have a next smallest number
     */
    public boolean hasNext() {
        return !linkedList.isEmpty();
    }
}
