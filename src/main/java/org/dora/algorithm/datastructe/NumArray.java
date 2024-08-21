package org.dora.algorithm.datastructe;

/**
 * mail lu.liu2@cariad-technology.cn
 * date 2024年07月21日
 * @author lu.liu2
 */
public class NumArray {


    private Node root = null;

    public NumArray(int[] nums) {
        root = internalConstructNode(nums, 0, nums.length - 1);
    }

    private Node internalConstructNode(int[] nums, int start, int end) {
        if (start == end) {
            Node node = new Node(start, end);
            node.sum = nums[start];
            return node;
        }
        int mid = start + (end - start) / 2;

        Node root = new Node(start, end);
        root.leftNode = internalConstructNode(nums, start, mid);
        root.rightNode = internalConstructNode(nums, mid + 1, end);
        root.sum = root.leftNode.sum + root.rightNode.sum;
        return root;
    }

    public int sumRange(int left, int right) {
        return this.sumHelper(this.root, left, right);

    }

    private int sumHelper(Node root, int left, int right) {
        if (left <= root.left && right >= root.right) {
            return root.sum;
        }
        if (right < root.left || left > root.right) {
            return 0;
        }
        return sumHelper(root.leftNode, left, right) + sumHelper(root.rightNode, left, right);
    }


    static class Node {
        private final int left;
        private final int right;

        private Node leftNode;

        private Node rightNode;

        private int sum;

        public Node(int left, int right) {
            this.left = left;
            this.right = right;
        }
    }


}

