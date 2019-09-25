package org.dora.algorithm.solution.v2;

import org.dora.algorithm.datastructe.TreeNode;

import java.util.LinkedList;
import java.util.Stack;

/**
 * @author dora
 * @date 2019/9/26
 */
public class BSTIterator {
    private LinkedList<Integer> ans;

    public BSTIterator(TreeNode root) {
        Stack<TreeNode> stack = new Stack<>();
        ans = new LinkedList<>();
        TreeNode p = root;
        while (!stack.isEmpty() || p != null) {
            while (p != null) {
                stack.push(p);
                p = p.left;
            }
            p = stack.pop();
            ans.add(p.val);

            p = p.right;
        }
    }

    /**
     * @return the next smallest number
     */
    public int next() {
        return ans.pollFirst();
    }

    /**
     * @return whether we have a next smallest number
     */
    public boolean hasNext() {
        return !ans.isEmpty();
    }



}
