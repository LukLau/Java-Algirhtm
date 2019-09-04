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


    /**
     * 105. Construct Binary Tree from Preorder and Inorder Traversal
     *
     * @param preorder
     * @param inorder
     * @return
     */
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        if (preorder == null || inorder == null) {
            return null;
        }
        return this.buildTree(0, preorder, 0, inorder.length - 1, inorder);
    }

    private TreeNode buildTree(int preStart, int[] preorder, int inStart, int inEnd, int[] inorder) {
        if (preStart >= preorder.length || inStart > inEnd) {
            return null;
        }

        TreeNode root = new TreeNode(preorder[preStart]);
        int index = 0;
        for (int i = inStart; i <= inEnd; i++) {
            if (inorder[i] == root.val) {
                index = i;
                break;
            }
        }
        root.left = this.buildTree(preStart + 1, preorder, inStart, index - 1, inorder);
        root.right = this.buildTree(preStart + index - inStart + 1, preorder, index + 1, inEnd, inorder);
        return root;
    }

    /**
     * 106. Construct Binary Tree from Inorder and Postorder Traversal
     *
     * @param inorder
     * @param postorder
     * @return
     */
    public TreeNode buildTreeII(int[] inorder, int[] postorder) {
        if (inorder == null || postorder == null) {
            return null;
        }
        return this.buildTreeII(0, inorder.length - 1, inorder, 0, postorder.length - 1, postorder);
    }

    private TreeNode buildTreeII(int inStart, int inEnd, int[] inorder, int postStart, int postEnd, int[] postorder) {
        if (inStart > inEnd || postStart > postEnd) {
            return null;
        }
        TreeNode root = new TreeNode(postorder[postEnd]);
        int index = 0;
        for (int i = inStart; i <= inEnd; i++) {
            if (inorder[i] == root.val) {
                index = i;
                break;
            }
        }
        root.left = this.buildTreeII(inStart, index - 1, inorder, postStart, postStart + index - inStart - 1, postorder);
        root.right = this.buildTreeII(index + 1, inEnd, inorder, postStart + index - inStart, postEnd - 1, postorder);
        return root;
    }

    /**
     * 107. Binary Tree Level Order Traversal II
     *
     * @param root
     * @return
     */
    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        if (root == null) {
            return Collections.emptyList();
        }
        LinkedList<List<Integer>> ans = new LinkedList<>();
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
            ans.addFirst(tmp);
        }
        return ans;
    }


    /**
     * 108. Convert Sorted Array to Binary Search Tree
     *
     * @param nums
     * @return
     */
    public TreeNode sortedArrayToBST(int[] nums) {
        if (nums == null || nums.length == 0) {
            return null;
        }
        return this.sortedArrayToBST(nums, 0, nums.length - 1);
    }

    private TreeNode sortedArrayToBST(int[] nums, int start, int end) {
        if (start > end) {
            return null;
        }
        int mid = start + (end - start) / 2;

        TreeNode root = new TreeNode(nums[mid]);
        root.left = this.sortedArrayToBST(nums, start, mid - 1);
        root.right = this.sortedArrayToBST(nums, mid + 1, end);
        return root;
    }

}
