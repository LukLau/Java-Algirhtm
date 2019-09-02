package org.dora.algorithm.solution.v2;

import org.dora.algorithm.datastructe.TreeNode;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

/**
 * @author dora
 * @date 2019/9/3
 */
public class TwoHundred {
    /**
     * 101. Symmetric Tree
     *
     * @param root
     * @return
     */
    public boolean isSymmetric(TreeNode root) {
        if (root == null) {
            return true;
        }
        return this.isSymmetric(root.left, root.right);
    }

    private boolean isSymmetric(TreeNode p, TreeNode q) {
        if (p == null && q == null) {
            return true;
        }
        if (p == null || q == null) {
            return false;
        }
        if (p.val == q.val) {
            return this.isSymmetric(p.left, q.right) && this.isSymmetric(p.right, q.left);
        }
        return false;
    }


    /**
     * 102. Binary Tree Level Order Traversal
     *
     * @param root
     * @return
     */
    public List<List<Integer>> levelOrder(TreeNode root) {
        if (root == null) {
            return Collections.emptyList();
        }
        List<List<Integer>> ans = new ArrayList<>();
        LinkedList<TreeNode> list = new LinkedList<>();
        list.add(root);
        while (!list.isEmpty()) {
            int size = list.size();
            List<Integer> tmp = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                TreeNode node = list.poll();
                tmp.add(node.val);
                if (node.left != null) {
                    list.add(node.left);
                }
                if (node.right != null) {
                    list.add(node.right);
                }
            }
            ans.add(tmp);
        }
        return ans;
    }


    /**
     * 103. Binary Tree Zigzag Level Order Traversal
     *
     * @param root
     * @return
     */
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        if (root == null) {
            return Collections.emptyList();
        }
        List<List<Integer>> ans = new ArrayList<>();
        LinkedList<TreeNode> list = new LinkedList<>();

        list.add(root);
        boolean leftToRight = true;
        while (!list.isEmpty()) {
            int size = list.size();
            LinkedList<Integer> tmp = new LinkedList<>();
            for (int i = 0; i < size; i++) {
                TreeNode node = list.poll();
                if (node.left != null) {
                    list.add(node.left);
                }
                if (node.right != null) {
                    list.add(node.right);
                }
                if (leftToRight) {
                    tmp.addLast(node.val);
                } else {
                    tmp.addFirst(node.val);
                }
            }
            leftToRight = !leftToRight;
            ans.add(tmp);
        }
        return ans;
    }

}
