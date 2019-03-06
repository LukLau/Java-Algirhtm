package org.dora.algorithm.solution;

import org.dora.algorithm.datastructe.TreeNode;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

/**
 * @author lauluk
 * @date 2019/03/07
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
            return false;
        }
        return isSymmetric(root.left, root.right);
    }

    private boolean isSymmetric(TreeNode left, TreeNode right) {
        if (left == null && right == null) {
            return true;
        }
        if (left == null || right == null) {
            return false;
        }
        if (left.val == right.val) {
            return isSymmetric(left.left, right.right) && isSymmetric(left.right, right.left);
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
            return new ArrayList<>();
        }
        List<List<Integer>> ans = new ArrayList<>();
        LinkedList<TreeNode> linkedList = new LinkedList<>();
        linkedList.addLast(root);
        while (!linkedList.isEmpty()) {
            int size = linkedList.size();
            List<Integer> tmp = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                TreeNode node = linkedList.pollFirst();
                tmp.add(node.val);
                if (node.left != null) {
                    linkedList.addLast(node.left);
                }
                if (node.right != null) {
                    linkedList.addLast(node.right);
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
            return new ArrayList<>();
        }
        LinkedList<TreeNode> list = new LinkedList<>();
        List<List<Integer>> ans = new ArrayList<>();
        list.addLast(root);
        boolean isLeft = false;

        while (!list.isEmpty()) {
            int size = list.size();
            List<Integer> tmp = new ArrayList<>();
            for (int i = 0; i < size; i++) {
            }
        }
        return ans;
    }

}
