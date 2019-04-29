package org.dora.algorithm.leetcode;

import org.dora.algorithm.datastructe.TreeNode;

/**
 * @author dora
 * @date 2019-04-29
 */
public class SecondPage {

    /**
     * 104. Maximum Depth of Binary Tree
     *
     * @param root
     * @return
     */
    public int maxDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        return 1 + Math.max(this.maxDepth(root.left), this.maxDepth(root.right));
    }

}
