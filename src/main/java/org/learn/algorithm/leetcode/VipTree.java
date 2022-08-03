package org.learn.algorithm.leetcode;

import org.learn.algorithm.datastructure.TreeNode;

public class VipTree {

    /**
     * 255. Verify Preorder Sequence in Binary Search Tree
     *
     * @param preorder: a string
     * @return: return a bool
     */
    public boolean isValidSerialization(String preorder) {
        // write your code here
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

        return internalCountUnival(root,);
    }

    private int internalCountUnival(TreeNode root, ) {
        if (root == null) {
            return 0;
        }
        if (root.left == null && root.right == null) {
            return 1;
        }
        int count = 0;

        if (root.)
    }


}
