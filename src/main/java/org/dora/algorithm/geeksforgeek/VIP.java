package org.dora.algorithm.geeksforgeek;

import org.dora.algorithm.datastructe.TreeNode;

/**
 * @author dora
 * @date 2019/11/29
 */
public class VIP {


    /**
     * 151 Binary Tree Upside Down Medium
     */
    public TreeNode upsizeDownBinaryTree(TreeNode root) {
        if (root == null || root.left == null) {
            return root;
        }
        TreeNode left = root.left;
        TreeNode right = root.right;
        TreeNode ans = this.upsizeDownBinaryTree(left);
        left.left = right;
        left.right = root;
        root.left = null;
        root.right = null;
        return ans;

    }
}
