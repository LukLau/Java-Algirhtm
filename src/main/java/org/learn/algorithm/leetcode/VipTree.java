package org.learn.algorithm.leetcode;

import org.learn.algorithm.datastructure.TreeNode;

import java.util.*;

public class VipTree {

    public static void main(String[] args) {
        VipTree vipTree = new VipTree();


        TreeNode root = new TreeNode(1);

        TreeNode node3 = new TreeNode(3);

        TreeNode node2 = new TreeNode(2);
        TreeNode node4 = new TreeNode(4);
        TreeNode node5 = new TreeNode(5);

        node3.left = node2;
        node3.right = node4;

        node4.right = node5;
        root.right = node3;

//        vipTree.longestConsecutiveii(root);


    }

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


    //  ---二叉树 最长连续序列问题----//


    /**
     * 298. Binary Tree Longest Consecutive Sequence
     *
     * @param root: the root of binary tree
     * @return: the length of the longest consecutive sequence path
     */


    public int longestConsecutive(TreeNode root) {
        // write your code here
        if (root == null) {
            return 0;
        }
//        LinkedList<TreeNode> linkedList = new LinkedList<>();
//
//        linkedList.offer(root);
//        int result = 0;
//        while (!linkedList.isEmpty()) {
//
//            TreeNode node = linkedList.poll();
//
//            int tmp = dfsLongestConsecutive(node, node.val);
//
//            result = Math.max(result, tmp);
//
//            if (node.left != null) {
//                linkedList.offer(node.left);
//            }
//            if (node.right != null) {
//                linkedList.offer(node.right);
//            }
//        }
//        return result;

        dfsLongestConsecutiveii(root, Integer.MIN_VALUE, 0);

        return longestConsecutive;
    }


    private int longestConsecutive = 0;

    private void dfsLongestConsecutiveii(TreeNode root, int val, int current) {
        if (root == null) {
            return;
        }
        if (root.val == val + 1) {
            current = current + 1;
        } else {
            current = 1;
        }
        longestConsecutive = Math.max(longestConsecutive, current);
        dfsLongestConsecutiveii(root.left, root.val, current);
        dfsLongestConsecutiveii(root.right, root.val, current);
    }


    /**
     * https://www.lintcode.com/problem/614
     *
     * @param root: the root of binary tree
     * @return: the length of the longest consecutive sequence path
     */
    public int longestConsecutive2(TreeNode root) {
        // write your code here
        if (root == null) {
            return 0;
        }
        LinkedList<TreeNode> linkedList = new LinkedList<>();
        linkedList.offer(root);
        while (!linkedList.isEmpty()) {
            TreeNode poll = linkedList.poll();

        }
//        intervalLongest(root,)
        return internalDFS(root, Integer.MIN_VALUE,0);
    }


}
