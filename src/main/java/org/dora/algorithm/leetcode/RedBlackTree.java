package org.dora.algorithm.leetcode;

import java.awt.*;

/**
 * date 2024年04月16日
 */
public class RedBlackTree {

    private final int RED_COLOR = 1;
    private final int BLACK_COLOR = 0;


    /**
     * 红黑树的特征
     * 1. 根节点是黑色的
     * 2. 红色节点的子节点是黑色的
     * 3. 非叶子节点不是黑色就是红色的
     * 4. 所有的叶子节点(null)都是黑色的
     * 5. 从任意节点出发 到其叶子节点的所有路径都包含相同数量的黑色子节点
     */

    private TreeNode root = null;

    public void add(TreeNode node) {
        TreeNode prev = null;
        TreeNode p = root;
        boolean leftChild = false;
        while (p != null) {
            prev = p;
            if (p.val > node.val) {
                p = p.left;
                leftChild = true;
            } else {
                p = p.right;
                leftChild = false;
            }
        }
        node.parent = prev;

        if (prev == null) {
            root = node;
            root.color = BLACK_COLOR;
            return;
        }
        if (leftChild) {
            prev.left = node;
        } else {
            prev.right = node;
        }
        balanceNode(node);
    }

    private void balanceNode(TreeNode node) {
        node.color = RED_COLOR;

        while (node.parent != null && node.parent.color != BLACK_COLOR) {
            TreeNode grandFather = node.parent.parent;

            if (grandFather.left == node.parent) {
                if (grandFather.right != null && grandFather.right.color == RED_COLOR) {
                    grandFather.right.color = BLACK_COLOR;
                    node.parent.color = BLACK_COLOR;

                    grandFather.color = RED_COLOR;
                    node = grandFather;
                    continue;
                } else {
                    if (node.parent.right == node) {
                        rightNode(node.parent);

                    }


                }
            }

        }

    }


    private void leftNode(TreeNode node) {
        if (node == null) {
            return;
        }
        TreeNode left = node.left;

        TreeNode parent = node.parent;

        parent.right = left;
        parent.parent = node;

        node.left = parent;
    }

    private void rightNode(TreeNode node) {
        if (node == null) {
            return;
        }

        TreeNode left = node.left;

        TreeNode right = left.right;

        left.right = node;
        node.parent = left;

        node.left = right;
        right.parent = node;


    }

    public static class TreeNode {


        public int val;

        public int color;
        public TreeNode parent;

        public TreeNode left;
        public TreeNode right;
    }

}
