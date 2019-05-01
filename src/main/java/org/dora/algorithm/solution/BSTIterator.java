package org.dora.algorithm.solution;

import org.dora.algorithm.datastructe.TreeNode;

import java.util.LinkedList;
import java.util.Stack;

/**
 * @author liulu
 * @date 2019-03-15
 */
public class BSTIterator {

    private LinkedList<Integer> ans = new LinkedList<>();

    public BSTIterator(TreeNode root) {
        if (root != null) {
            Stack<TreeNode> stack = new Stack<>();
            TreeNode node = root;
            while (!stack.isEmpty() || node != null) {
                while (node != null) {
                    stack.push(node);
                    node = node.left;
                }
                node = stack.pop();
                ans.add(node.val);
                node = node.right;
            }
        }
    }

    /**
     * @return the next smallest number
     */
    public int next() {
        return ans.pop();
    }

    /**
     * @return whether we have a next smallest number
     */
    public boolean hasNext() {
        return !ans.isEmpty();
    }
}
