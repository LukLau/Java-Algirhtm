package org.learn.algorithm.leetcode;

import org.learn.algorithm.datastructure.ListNode;
import org.learn.algorithm.datastructure.Node;
import org.learn.algorithm.datastructure.TreeNode;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Stack;

/**
 * 树的解决方案
 *
 * @author luk
 * @date 2021/4/10
 */
public class TreeSolution {
    /**
     * 124. Binary Tree Maximum Path Sum
     *
     * @param root
     * @return
     */
    private int maxPathSum = Integer.MIN_VALUE;

    // 判断树是不是满足标准

    public static void main(String[] args) {
        TreeSolution solution = new TreeSolution();

        ListNode root = new ListNode(-1);

        ListNode n1 = new ListNode(0);

        root.next = n1;

        n1.next = new ListNode(1);

        n1.next.next = new ListNode(2);


        solution.sortedListToBST(root);
    }


    // 排序系列//

    public int minDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int left = minDepth(root.left);

        int right = minDepth(root.right);

        return 1 + Math.min(left, right);
    }

    /**
     * 类似于归并排序
     * todo 插入排序
     * 147. Insertion Sort List
     *
     * @param head
     * @return
     */
    public ListNode insertionSortList(ListNode head) {

        return null;
    }

    /**
     * 148. Sort List
     *
     * @param head
     * @return
     */
    public ListNode sortList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode slow = head;
        ListNode fast = head;
        while (fast.next != null && fast.next.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode tmp = slow.next;
        slow.next = null;
        ListNode l1 = sortList(head);
        ListNode l2 = sortList(tmp);
        return merge(l1, l2);
    }

    private ListNode merge(ListNode first, ListNode second) {
        if (first == null && second == null) {
            return null;
        }
        if (first == null) {
            return second;
        }
        if (second == null) {
            return first;
        }
        if (first.val <= second.val) {
            first.next = merge(first.next, second);
            return first;
        } else {
            second.next = merge(first, second.next);
            return second;
        }
    }

    /**
     * 94. Binary Tree Inorder Traversal
     *
     * @param root
     * @return
     */
    public List<Integer> inorderTraversal(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        List<Integer> result = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        TreeNode p = root;
        while (!stack.isEmpty() || p != null) {
            while (p != null) {
                stack.push(p);
                p = p.left;
            }
            p = stack.pop();
            result.add(p.val);
            p = p.right;
        }
        return result;
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
        List<List<Integer>> result = new ArrayList<>();
        boolean leftToRight = true;
        LinkedList<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            LinkedList<Integer> tmp = new LinkedList<>();
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode poll = queue.poll();
                if (leftToRight) {
                    tmp.addLast(poll.val);
                } else {
                    tmp.addFirst(poll.val);
                }
                if (poll.left != null) {
                    queue.offer(poll.left);
                }
                if (poll.right != null) {
                    queue.offer(poll.right);
                }
            }
            leftToRight = !leftToRight;
            result.add(tmp);
        }
        return result;
    }

    /**
     * 107. Binary Tree Level Order Traversal II
     *
     * @param root
     * @return
     */
    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        LinkedList<TreeNode> linkedList = new LinkedList<>();
        linkedList.offer(root);
        List<List<Integer>> result = new ArrayList<>();
        while (!linkedList.isEmpty()) {
            int size = linkedList.size();
            List<Integer> tmp = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                TreeNode poll = linkedList.poll();
                tmp.add(poll.val);
                if (poll.left != null) {
                    linkedList.offer(poll.left);
                }
                if (poll.right != null) {
                    linkedList.offer(poll.right);
                }
            }
            result.add(tmp);
        }
        return result;
    }

    /**
     * 145. Binary Tree Postorder Traversal
     *
     * @param root
     * @return
     */

    public List<Integer> postorderTraversal(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        LinkedList<Integer> result = new LinkedList<>();
        Stack<TreeNode> stack = new Stack<>();
        TreeNode p = root;
        while (!stack.isEmpty() || p != null) {
            if (p != null) {
                stack.push(p);
                result.addFirst(p.val);
                p = p.right;
            } else {
                p = stack.pop();
                p = p.left;
            }
        }
        return result;
    }

    /**
     * 199. Binary Tree Right Side View
     *
     * @param root
     * @return
     */
    public List<Integer> rightSideView(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        List<Integer> result = new ArrayList<>();
        intervalRightSideView(result, root, 0);
        return result;
    }

    private void intervalRightSideView(List<Integer> result, TreeNode root, int currentLevel) {
        if (root == null) {
            return;
        }
        if (result.size() == currentLevel) {
            result.add(root.val);
        }
        intervalRightSideView(result, root.right, currentLevel + 1);
        intervalRightSideView(result, root.left, currentLevel + 1);
    }

    /**
     * 230. Kth Smallest Element in a BST
     *
     * @param root
     * @param k
     * @return
     */
    public int kthSmallest(TreeNode root, int k) {
        if (root == null) {
            return -1;
        }
        List<Integer> result = inorderTraversal(root);
        return result.get(k - 1);
    }

    public int kthSmallestV2(TreeNode root, int k) {
        if (root == null) {
            return -1;
        }
        k--;
        Stack<TreeNode> stack = new Stack<>();
        TreeNode q = root;
        int iterator = 0;
        while (!stack.isEmpty() || q != null) {
            while (q != null) {
                stack.push(q);
                q = q.left;
            }
            q = stack.pop();
            if (iterator == k) {
                return q.val;
            }
            iterator++;
            q = q.right;
        }
        return -1;
    }

    /**
     * 285
     * Inorder Successor in BST
     *
     * @param root: The root of the BST.
     * @param p:    You need find the successor node of p.
     * @return: Successor of p.
     */
    public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
        // write your code here
        Stack<TreeNode> stack = new Stack<>();
        TreeNode q = root;
        TreeNode prev = null;
        while (!stack.isEmpty() || q != null) {
            while (q != null) {
                stack.push(q);
                q = q.left;
            }
            q = stack.pop();
            if (prev != null && prev == p) {
                return q;
            }
            prev = q;

            q = q.right;
        }
        return null;
    }

    /**
     * todo
     *
     * @param root
     * @param p
     * @return
     */
    public TreeNode inorderSuccessorV2(TreeNode root, TreeNode p) {
        if (root == null || p == null) {
            return null;
        }
        return intervalNode(root, p);
    }


    // --生成树系列 //

    private TreeNode intervalNode(TreeNode root, TreeNode p) {
        if (root == null) {
            return null;
        }
        if (root == p) {
            p = p.right;
            while (p.left != null) {
                p = p.left;
            }
            return p;
        }
        if (root.left == p) {
            return root;
        }
        if (root.right == p) {
            return root;
        }
        TreeNode node = intervalNode(root.left, p);
        if (node != null) {
            return node;
        }
        TreeNode rightNode = intervalNode(root.right, p);
        if (rightNode != null) {
            return root;
        }
        return null;
    }

    /**
     * 95. Unique Binary Search Trees II
     *
     * @param n
     * @return
     */
    public List<TreeNode> generateTrees(int n) {
        if (n <= 0) {
            return new ArrayList<>();
        }
        return intervalGenerateTrees(1, n);
    }

    private List<TreeNode> intervalGenerateTrees(int start, int end) {
        List<TreeNode> result = new ArrayList<>();
        if (start > end) {
            result.add(null);
            return result;
        }
        if (start == end) {
            result.add(new TreeNode(start));
            return result;
        }
        for (int i = start; i <= end; i++) {
            List<TreeNode> leftNodes = intervalGenerateTrees(start, i - 1);
            List<TreeNode> rightNodes = intervalGenerateTrees(i + 1, end);
            for (TreeNode leftNode : leftNodes) {
                for (TreeNode rightNode : rightNodes) {
                    TreeNode root = new TreeNode(i);
                    root.left = leftNode;
                    root.right = rightNode;
                    result.add(root);
                }
            }
        }
        return result;
    }


    // 二叉搜索树相关//

    /**
     * 96. Unique Binary Search Trees
     *
     * @param n
     * @return
     */
    public int numTrees(int n) {
        if (n <= 1) {
            return n;
        }
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            for (int j = 1; j <= i; j++) {
                dp[i] += dp[j - 1] * dp[i - j];
            }
        }
        return dp[n];
    }

    /**
     * 98. Validate Binary Search Tree
     *
     * @param root
     * @return
     */
    public boolean isValidBST(TreeNode root) {
        if (root == null) {
            return false;
        }
        Stack<TreeNode> stack = new Stack<>();
        TreeNode prev = null;
        TreeNode p = root;
        while (!stack.isEmpty() || p != null) {
            while (p != null) {
                stack.push(p);
                p = p.left;
            }
            p = stack.pop();
            if (prev != null && prev.val >= p.val) {
                return false;
            }
            prev = p;
            p = p.right;
        }
        return true;
    }

    public boolean isValidBSTV2(TreeNode root) {
        if (root == null) {
            return false;
        }
        return intervalValidBST(Long.MIN_VALUE, root, Long.MAX_VALUE);
    }

    private boolean intervalValidBST(long minValue, TreeNode root, long maxValue) {
        if (root == null) {
            return true;
        }
        if (root.val <= minValue || root.val >= maxValue) {
            return false;
        }
        return intervalValidBST(minValue, root.left, root.val) && intervalValidBST(root.val, root.right, maxValue);
    }

    /**
     * 99. Recover Binary Search Tree
     *
     * @param root
     */
    public void recoverTree(TreeNode root) {
        if (root == null) {
            return;
        }
        Stack<TreeNode> stack = new Stack<>();
        TreeNode prev = null;
        TreeNode p = root;
        TreeNode first = null;
        TreeNode second = null;
        while (!stack.isEmpty() || p != null) {
            while (p != null) {
                stack.push(p);
                p = p.left;
            }
            p = stack.pop();

            if (prev != null) {
                if (first == null && prev.val >= p.val) {
                    first = prev;
                }
                if (first != null && prev.val >= p.val) {
                    second = p;
                }
            }
            prev = p;
            p = p.right;
        }
        if (first != null) {
            int val = first.val;
            first.val = second.val;
            second.val = val;
        }
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
        return intervalSorted(nums, 0, nums.length - 1);
    }

    private TreeNode intervalSorted(int[] nums, int start, int end) {
        if (start > end) {
            return null;
        }
        int mid = start + (end - start) / 2;
        TreeNode root = new TreeNode(nums[mid]);
        root.left = intervalSorted(nums, start, mid - 1);
        root.right = intervalSorted(nums, mid + 1, end);
        return root;
    }


    // 同一颗树//

    /**
     * 109. Convert Sorted List to Binary Search Tree
     *
     * @param head
     * @return
     */
    public TreeNode sortedListToBST(ListNode head) {
        if (head == null) {
            return null;
        }
        if (head.next == null) {
            return new TreeNode(head.val);
        }
        ListNode fast = head;
        ListNode slow = head;
        ListNode prev = head;
        while (fast != null && fast.next != null) {
            prev = slow;
            slow = slow.next;
            fast = fast.next.next;
        }
        prev.next = null;
        TreeNode root = new TreeNode(slow.val);
        root.left = sortedListToBST(head);
        ListNode next = slow.next;
        slow.next = null;
        root.right = sortedListToBST(next);
        return root;
    }

    /**
     * 100. Same Tree
     *
     * @param p
     * @param q
     * @return
     */
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null && q == null) {
            return true;
        }
        if (p == null || q == null) {
            return false;
        }
        if (p.val != q.val) {
            return false;
        }
        return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
    }

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
        return intervalSymmetric(root.left, root.right);
    }


    private boolean intervalSymmetric(TreeNode left, TreeNode right) {
        if (left == null && right == null) {
            return true;
        }
        if (left == null || right == null) {
            return false;
        }
        if (left.val != right.val) {
            return false;
        }
        return intervalSymmetric(left.left, right.right) && intervalSymmetric(left.right, right.left);
    }

    // --构造二叉树 //


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
        return intervalBuildTree(0, preorder, 0, inorder.length - 1, inorder);
    }

    private TreeNode intervalBuildTree(int preStart, int[] preorder, int inStart, int inEnd, int[] inorder) {
        if (preStart >= preorder.length || inStart > inEnd) {
            return null;
        }
        int index = 0;
        TreeNode root = new TreeNode(preorder[preStart]);
        for (int i = inStart; i <= inEnd; i++) {
            if (inorder[i] == root.val) {
                index = i;
                break;
            }
        }
        root.left = intervalBuildTree(preStart + 1, preorder, inStart, index - 1, inorder);
        root.right = intervalBuildTree(preStart + index - inStart + 1, preorder, index + 1, inEnd, inorder);
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
        if (inorder == null || inorder.length == 0 || postorder == null || postorder.length == 0) {
            return null;
        }
        return intervalBuildTree(0, inorder.length - 1, inorder, 0, postorder.length - 1, postorder);
    }


    private TreeNode intervalBuildTree(int inStart, int inEnd, int[] inorder, int postStart, int postEnd, int[] postorder) {
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
        root.left = intervalBuildTree(inStart, index - 1, inorder, postStart, postStart + index - inStart - 1, postorder);
        root.right = intervalBuildTree(index + 1, inEnd, inorder, postStart + index - inStart, postEnd - 1, postorder);
        return root;
    }


    // 公共祖先lca


    /**
     * 235. Lowest Common Ancestor of a Binary Search Tree
     *
     * @param root
     * @param p
     * @param q
     * @return
     */
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == p || root == q) {
            return root;
        }
        if (root.val > p.val && root.val > q.val) {
            return lowestCommonAncestor(root.left, p, q);
        } else if (root.val < p.val && root.val < q.val) {
            return lowestCommonAncestor(root.right, p, q);
        } else {
            return root;
        }
    }


    /**
     * NC102 在二叉树中找到两个节点的最近公共祖先
     *
     * @param root TreeNode类
     * @param o1   int整型
     * @param o2   int整型
     * @return int整型
     */
    public int lowestCommonAncestor(TreeNode root, int o1, int o2) {
        // write code here
        if (root == null) {
            return -1;
        }
        if (root.val == o1 || root.val == o2) {
            return root.val;
        }
        int left = lowestCommonAncestor(root.left, o1, o2);
        int right = lowestCommonAncestor(root.right, o1, o2);
        if (left != -1 && right != -1) {
            return root.val;
        } else if (left == -1) {
            return right;
        } else {
            return left;
        }
    }


    /**
     * 236. Lowest Common Ancestor of a Binary Tree
     *
     * @param root
     * @param p
     * @param q
     * @return
     */
    public TreeNode lowestCommonAncestorII(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == p || root == q) {
            return root;
        }
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if (left != null && right != null) {
            return root;
        } else if (left != null) {
            return left;
        } else {
            return right;
        }
    }

    /**
     * 257. Binary Tree Paths
     *
     * @param root
     * @return
     */
    public List<String> binaryTreePaths(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        List<String> result = new ArrayList<>();
        intervalBinaryTree(result, "", root);
        return result;
    }

    // --- //
    private void intervalBinaryTree(List<String> result, String s, TreeNode root) {
        if (root == null) {
            return;
        }
        if (s.isEmpty()) {
            s = s + root.val;
        } else {
            s = s + "->" + root.val;
        }
        if (root.left == null && root.right == null) {
            result.add(s);
            return;
        }
        intervalBinaryTree(result, s, root.left);
        intervalBinaryTree(result, s, root.right);
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
        return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
    }

    /**
     * 112. Path Sum
     *
     * @param root
     * @param targetSum
     * @return
     */
    public boolean hasPathSum(TreeNode root, int targetSum) {
        if (root == null) {
            return false;
        }
        if (root.left == null && root.right == null && root.val == targetSum) {
            return true;
        }
        return hasPathSum(root.left, targetSum - root.val) || hasPathSum(root.right, targetSum - root.val);
    }

    /**
     * 113. Path Sum II
     *
     * @param root
     * @param targetSum
     * @return
     */
    public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
        if (root == null) {
            return new ArrayList<>();
        }
        List<List<Integer>> result = new ArrayList<>();
        intervalPathSum(result, new ArrayList<>(), root, targetSum);
        return result;
    }

    private void intervalPathSum(List<List<Integer>> result, List<Integer> tmp, TreeNode root, int targetSum) {
        tmp.add(root.val);
        if (root.left == null && root.right == null && root.val == targetSum) {
            result.add(new ArrayList<>(tmp));
        } else {
            if (root.left != null) {
                intervalPathSum(result, tmp, root.left, targetSum - root.val);
            }
            if (root.right != null) {
                intervalPathSum(result, tmp, root.right, targetSum - root.val);
            }
        }
        tmp.remove(tmp.size() - 1);
    }

    /**
     * 114. Flatten Binary Tree to Linked List
     *
     * @param root
     */
    public void flatten(TreeNode root) {
        if (root == null) {
            return;
        }
        Stack<TreeNode> stack = new Stack<>();
        TreeNode p = root;
        TreeNode prev = null;
        stack.push(p);
        while (!stack.isEmpty()) {
            TreeNode pop = stack.pop();
            if (pop.right != null) {
                stack.push(pop.right);
            }
            if (pop.left != null) {
                stack.push(pop.left);
            }
            if (prev != null) {
                prev.right = pop;
                prev.left = null;
            }
            prev = pop;
        }
    }

    /**
     * 116. Populating Next Right Pointers in Each Node
     *
     * @param root
     * @return
     */
    public Node connect(Node root) {
        if (root == null || root.left == null) {
            return root;
        }
        Node current = root;
        while (current.left != null) {
            Node tmp = current.left;
            while (current != null) {
                current.left.next = current.right;
                if (current.next != null) {
                    current.right.next = current.next.left;
                }
                current = current.next;
            }
            current = tmp;
        }
        return root;
    }

    /**
     * 117. Populating Next Right Pointers in Each Node II
     *
     * @param root
     * @return
     */
    public Node connectII(Node root) {
        if (root == null) {
            return null;
        }
        Node current = root;
        while (current != null) {
            Node head = null;
            Node prev = null;
            while (current != null) {
                if (current.left != null) {
                    if (head == null) {
                        head = current.left;
                    } else {
                        prev.next = current.left;
                    }
                    prev = current.left;
                }
                if (current.right != null) {
                    if (head == null) {
                        head = current.right;
                    } else {
                        prev.next = current.right;
                    }
                    prev = current.right;
                }
                current = current.next;
            }
            current = head;
        }
        return root;
    }

    public int maxPathSum(TreeNode root) {
        if (root == null) {
            return 0;
        }
        intervalPathSum(root);
        return maxPathSum;
    }

    private int intervalPathSum(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int left = intervalPathSum(root.left);
        int right = intervalPathSum(root.right);
        left = Math.max(left, 0);
        right = Math.max(right, 0);
        maxPathSum = Math.max(maxPathSum, root.val + left + right);
        return Math.max(left, right) + root.val;
    }


    /**
     * 129. Sum Root to Leaf Numb
     *
     * @param root
     * @return
     */
    public int sumNumbers(TreeNode root) {
        if (root == null) {
            return 0;
        }
        return intervalSumNumbers(root, 0);
    }

    private int intervalSumNumbers(TreeNode root, int val) {
        if (root == null) {
            return 0;
        }
        if (root.left == null && root.right == null) {
            return val * 10 + root.val;
        }
        return intervalSumNumbers(root.left, val * 10 + root.val)
                + intervalSumNumbers(root.right, val * 10 + root.val);
    }


}
