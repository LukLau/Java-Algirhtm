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

    // 排序系列//

    /**
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
        ListNode fast = head;
        ListNode slow = head;
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        ListNode next = slow.next;

        slow.next = null;

        ListNode first = sortList(head);

        ListNode second = sortList(next);

        return merge(first, second);
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

    // 树的遍历//

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
        LinkedList<TreeNode> deque = new LinkedList<>();

        LinkedList<List<Integer>> result = new LinkedList<>();

        deque.offer(root);
        while (!deque.isEmpty()) {
            int size = deque.size();
            List<Integer> tmp = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                TreeNode node = deque.poll();
                tmp.add(node.val);
                if (node.left != null) {
                    deque.offer(node.left);
                }
                if (node.right != null) {
                    deque.offer(node.right);
                }
            }
            result.addFirst(tmp);
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
        List<Integer> result = new ArrayList<>();

        intervalRightSideView(root, result, 0);

        return result;
    }

    private void intervalRightSideView(TreeNode root, List<Integer> result, int currentLevel) {
        if (root == null) {
            return;
        }
        if (result.size() == currentLevel) {
            result.add(root.val);
        }
        intervalRightSideView(root.right, result, currentLevel + 1);

        intervalRightSideView(root.left, result, currentLevel + 1);
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


    // --生成树系列 //


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
        if (start == end) {
            result.add(new TreeNode(start));
            return result;
        }
        if (start > end) {
            result.add(null);
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


    /**
     * 96. Unique Binary Search Trees
     *
     * @param n
     * @return
     */
    public int numTrees(int n) {
        if (n <= 0) {
            return 0;
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


    // 二叉搜索树相关//

    /**
     * 98. Validate Binary Search Tree
     *
     * @param root
     * @return
     */
    public boolean isValidBST(TreeNode root) {
        if (root == null) {
            return true;
        }
        Stack<TreeNode> stack = new Stack<>();
        TreeNode p = root;
        TreeNode prev = null;
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


    /**
     * 99. Recover Binary Search Tree
     *
     * @param root
     */
    public void recoverTree(TreeNode root) {
        if (root == null) {
            return;
        }
        TreeNode first = null;

        TreeNode second = root;
        Stack<TreeNode> stack = new Stack<>();
        TreeNode p = root;
        TreeNode prev = null;
        while (!stack.isEmpty() || p != null) {
            while (p != null) {
                stack.push(p);
                p = p.left;
            }
            p = stack.pop();
            if (prev != null && prev.val >= p.val) {
                if (first == null) {
                    first = prev;
                }
                second = p;
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


    /**
     * 109. Convert Sorted List to Binary Search Tree
     *
     * @param head
     * @return
     */
    public TreeNode sortedListToBST(ListNode head) {
        if (head == null || head.next == null) {
            return head == null ? null : new TreeNode(head.val);
        }
        ListNode fast = head;
        ListNode slow = head;
        ListNode prev = slow;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            prev = slow;
            slow = slow.next;
        }
        TreeNode root = new TreeNode(slow.val);

        prev.next = null;

        root.left = sortedListToBST(head);

        root.right = sortedListToBST(slow.next);

        return root;
    }


    // 同一颗树//


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
        return intervalSymmetric(root.left, root.right) && intervalSymmetric(root.right, root.left);
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
        TreeNode root = new TreeNode(preorder[preStart]);
        int index = 0;
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
        if (inorder == null || postorder == null) {
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
        if (root == null || root == q || root == p) {
            return root;
        }
        if (p.val < root.val && q.val < root.val) {
            return lowestCommonAncestor(root.left, p, q);
        } else if (p.val > root.val && q.val > root.val) {
            return lowestCommonAncestor(root.right, p, q);
        }
        return root;
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
        if (root == null || root == q || root == p) {
            return root;
        }
        TreeNode left = lowestCommonAncestorII(root.left, p, q);

        TreeNode right = lowestCommonAncestorII(root.right, p, q);
        if (left != null && right != null) {
            return root;
        } else if (left == null && right != null) {
            return right;
        } else {
            return left;
        }
    }

    // --- //


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
        } else {
            if (root.left != null && hasPathSum(root.left, targetSum - root.val)) {
                return true;
            }
            return root.right != null && hasPathSum(root.right, targetSum - root.val);
        }
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
        if (root == null) {
            return null;
        }
        Node current = root;
        while (current.left != null) {
            Node nextLevel = current.left;
            while (current != null) {
                current.left.next = current.right;
                if (current.next != null) {
                    current.right.next = current.next.left;
                }
                current = current.next;
            }
            current = nextLevel;
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


    /**
     * 124. Binary Tree Maximum Path Sum
     *
     * @param root
     * @return
     */
    private int maxPathSum = Integer.MIN_VALUE;

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
        int val = left + right + root.val;
        maxPathSum = Math.max(val, maxPathSum);
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
        int result = val * 10 + root.val;
        if (root.left == null && root.right == null) {
            return result;
        }
        return intervalSumNumbers(root.left, result) + intervalSumNumbers(root.right, result);
    }


}
