package org.learn.algorithm.leetcode;

import org.learn.algorithm.datastructure.TreeNode;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Stack;

public class VipTree {

    /**
     * 255. Verify Preorder Sequence in Binary Search Tree
     *
     * @param preorder: a string
     * @return: return a bool
     */
    public boolean isValidSerialization(String preorder) {
        // write your code here
        return false;
    }


    /**
     * 250. Count Univalue Subtrees
     *
     * @param root
     * @return
     */
    public int countUnivalSubtrees(TreeNode root) {
        // write your code here
        if (root == null) {
            return 0;
        }
        int count = 0;
        if (isUnivalTree(root, root.val)) {
            count++;
        }
        count += countUnivalSubtrees(root.left);
        count += countUnivalSubtrees(root.right);
        return count;
    }

    private boolean isUnivalTree(TreeNode root, int val) {
        if (root == null) {
            return true;
        }
        return root.val == val && isUnivalTree(root.left, root.val) && isUnivalTree(root.right, root.val);
    }


    /**
     * 270. Closest Binary Search Tree Val
     *
     * @param root:   the given BST
     * @param target: the given target
     * @return: the value in the BST that is closest to the target
     */
    Integer result = null;

    public int closestValue(TreeNode root, double target) {
        // write your code here

        internalClose(root, target);

        return result;
    }

    private void internalClose(TreeNode root, double target) {
        if (root == null) {
            return;
        }
        if (root.val == target) {
            result = (int) target;
            return;
        }
        if (result == null || Math.abs(result - target) > Math.abs(root.val - target)) {
            result = root.val;
        }
        if (root.val < target) {
            internalClose(root.right, target);
        } else {
            internalClose(root.left, target);
        }
    }


    /**
     * 272. Closest Binary Search Tree Val
     *
     * @param root:   the given BST
     * @param target: the given target
     * @param k:      the given k
     * @return: k values in the BST that are closest to the target
     * we will sort your return value in output
     */
    public List<Integer> closestKValues(TreeNode root, double target, int k) {
        // write your code here
        if (root == null) {
            return new ArrayList<>();
        }
        Stack<TreeNode> stack = new Stack<>();

        stack.push(root);
        while (!stack.isEmpty()) {
            TreeNode pop = stack.pop();
        }
        return null;
    }


    /**
     * #285 Inorder Successor in BST
     *
     * @param root: The root of the BST.
     * @param p:    You need find the successor node of p.
     * @return: Successor of p.
     */
    public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
        // write your code here
        if (root == null) {
            return null;
        }
        Stack<TreeNode> stack = new Stack<>();
        TreeNode prev = root;
        while (!stack.isEmpty() || prev != null) {
            while (prev != null) {
                stack.push(prev);
                prev = prev.left;
            }
            prev = stack.pop();
            if (prev == p) {
                return prev.right;
            }
            prev = prev.right;
        }
        return null;


    }


}
