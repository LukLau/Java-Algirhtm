package org.learn.algorithm.leetcode;

import org.learn.algorithm.datastructure.TreeNode;

/**
 * Vip 题目
 *
 * @author luk
 * @date 2021/4/12
 */
public class VipSolution {

    /**
     * 156 Binary Tree Upside Down
     *
     * @param root: the root of binary tree
     * @return: new root
     */
    public TreeNode upsideDownBinaryTree(TreeNode root) {
        if (root == null || root.left == null) {
            return root;
        }
        TreeNode node = upsideDownBinaryTree(root.left);

        root.left.left = root.right;

        root.left.right = root;

        root.left = null;

        root.right = null;

        return node;
        // write your code here
    }


    /**
     * 161 One Edit Distance
     *
     * @param s: a string
     * @param t: a string
     * @return: true if they are both one edit distance apart or false
     */
    public boolean isOneEditDistance(String s, String t) {
        if (s == null || t == null) {
            return false;
        }
        if (s.equals(t)) {
            return false;
        }
        int len = Math.min(s.length(), t.length());

        int diffCount = 0;
        for (int i = 0; i < len; i++) {
            if (s.charAt(i) != t.charAt(i)) {
                return s.substring(i).equals(t);
            }
        }

        // write your code here
    }


}
