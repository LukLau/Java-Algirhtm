package org.learn.algorithm.leetcode;

import org.learn.algorithm.datastructure.ListNode;
import org.learn.algorithm.datastructure.Node;
import org.learn.algorithm.datastructure.TreeNode;

import java.util.*;

/**
 * 树的解决方案
 *
 * @author luk
 * @date 2021/4/10
 */
public class TreeSolution {

    // 二叉树遍历相关

    /**
     * 124. Binary Tree Maximum Path Sum
     *
     * @param root
     * @return
     */
    private int maxPathSum = Integer.MIN_VALUE;

    public static void main(String[] args) {
        TreeSolution solution = new TreeSolution();

//        TreeNode root = new TreeNode(1);
//        TreeNode left = new TreeNode(2);
//        left.left = new TreeNode(4);
//        root.left = left;
//        root.right = new TreeNode(3);
//
//        left.right = new TreeNode(5);

        TreeNode root = new TreeNode(Integer.MIN_VALUE);

        root.right = new TreeNode(Integer.MAX_VALUE);


//        solution.binaryTreePaths(root);
        solution.isValidBSTV2(root);
    }

    /**
     * https://www.lintcode.com/problem/1307/
     *
     * @param preorder: List[int]
     * @return: return a boolean
     */
    public boolean verifyPreorder(int[] preorder) {
        // write your code here
        if (preorder == null || preorder.length == 0) {
            return false;
        }
        Stack<Integer> stack = new Stack<>();
        Integer prev = null;
        for (int currentNodeValue : preorder) {
            if (prev != null && currentNodeValue <= prev) {
                return false;
            }

            while (!stack.isEmpty() && stack.peek() < currentNodeValue) {
                prev = stack.pop();
            }
            stack.push(currentNodeValue);
        }
        return true;
    }

    // 判断树是不是满足标准

    public boolean verifyPreorderii(int[] preorder) {
        if (preorder == null || preorder.length == 0) {
            return true;
        }
        int index = -1;
        Integer prev = null;
        for (int i = 0; i < preorder.length; i++) {
            int val = preorder[i];
            if (prev != null && prev <= val) {
                return false;
            }
            while (index >= 0 && preorder[index] < val) {
                prev = preorder[index--];
            }
            preorder[++index] = val;
        }
        return true;
    }


    public int minDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        if (root.left == null) {
            return 1 + minDepth(root.right);
        }
        if (root.right == null) {
            return 1 + minDepthii(root.left);
        }
        int left = minDepth(root.left);

        int right = minDepth(root.right);

        return Math.min(left, right) + 1;
    }


    public int minDepthii(TreeNode root) {
        if (root == null) {
            return 0;
        }
        if (root.left == null && root.right == null) {
            return 1;
        }
        if (root.left == null || root.right == null) {
            return root.left == null ? 1 + minDepth(root.right) : 1 + minDepth(root.left);
        }
        int left = minDepth(root.left);
        int right = minDepth(root.right);
        return Math.min(left, right) + 1;
    }

    // 排序系列//


    /**
     * 类似于归并排序
     * todo 插入排序
     * 147. Insertion Sort List
     *
     * @param head
     * @return
     */
    public ListNode insertionSortList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode dummy = new ListNode(0);

        ListNode prev = dummy;

        while (head != null) {
            ListNode tmp = head.next;
            if (prev.val >= head.val) {
                prev = dummy;
            }
        }
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
        ListNode l1 = sortList(head);
        ListNode l2 = sortList(next);
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
        LinkedList<TreeNode> linkedList = new LinkedList<>();
        linkedList.offer(root);
        boolean leftToRight = true;
        while (!linkedList.isEmpty()) {
            int size = linkedList.size();
            LinkedList<Integer> tmp = new LinkedList<>();
            for (int i = 0; i < size; i++) {
                TreeNode node = linkedList.poll();
                if (node.left != null) {
                    linkedList.offer(node.left);
                }
                if (node.right != null) {
                    linkedList.offer(node.right);
                }
                if (leftToRight) {
                    tmp.addLast(node.val);
                } else {
                    tmp.addFirst(node.val);
                }
            }
            result.add(tmp);
            leftToRight = !leftToRight;
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
        LinkedList<List<Integer>> result = new LinkedList<>();
        LinkedList<TreeNode> linkedList = new LinkedList<>();
        linkedList.offer(root);
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
        LinkedList<Integer> linkedList = new LinkedList<>();
        Stack<TreeNode> stack = new Stack<>();
        TreeNode p = root;
        while (!stack.isEmpty() || p != null) {
            if (p != null) {
                linkedList.addFirst(p.val);
                stack.push(p);
                p = p.right;
            } else {
                p = stack.pop();
                p = p.left;
            }
        }
        return linkedList;
    }


    public int[] postorderTraversalII(TreeNode root) {
        if (root == null) {
            return new int[]{};
        }
        LinkedList<Integer> linkedList = new LinkedList<>();
        Stack<TreeNode> stack = new Stack<>();
        TreeNode p = root;
        while (p != null || !stack.isEmpty()) {
            if (p != null) {
                stack.push(p);
                linkedList.addFirst(p.val);
                p = p.right;
            } else {
                p = stack.pop();
                p = p.left;
            }
        }
        int[] result = new int[linkedList.size()];
        int index = 0;
        for (int i = 0; i < result.length; i++) {
            result[index] = linkedList.get(index);
            index++;
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
        internalRightSideView(result, 0, root);
        return result;
    }

    private void internalRightSideView(List<Integer> result, int expected, TreeNode root) {
        if (root == null) {
            return;
        }
        if (result.size() == expected) {
            result.add(root.val);
        }
        expected++;
        internalRightSideView(result, expected, root.right);
        internalRightSideView(result, expected, root.left);
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
     * trick: 类似二分查询思想
     *
     * @param root
     * @param p
     * @return
     */
    public TreeNode inorderSuccessorii(TreeNode root, TreeNode p) {
        if (root == null) {
            return null;
        }
        if (root.val <= p.val) {
            return inorderSuccessor(root.right, p);
        }
        TreeNode successor = inorderSuccessor(root.left, p);
        if (successor == null) {
            return root;
        }
        return successor;
    }

    public TreeNode inorderSuccessoriii(TreeNode root, TreeNode p) {
        if (root == null) {
            return null;
        }
        TreeNode candidate = null;
        while (root != null) {
            if (p.val >= root.val) {
                root = root.right;
            } else {
                candidate = root;
                root = root.left;
            }
        }
        return candidate;
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
        return internalGenerateTress(1, n);
    }

    private List<TreeNode> internalGenerateTress(int start, int end) {
        List<TreeNode> result = new ArrayList<>();
        if (start > end) {
            result.add(null);
            return result;
        }
        if (start == end) {
            TreeNode node = new TreeNode(start);
            result.add(node);
            return result;
        }
        for (int i = start; i <= end; i++) {
            List<TreeNode> leftNodes = internalGenerateTress(start, i - 1);

            List<TreeNode> rightNodes = internalGenerateTress(i + 1, end);

            for (TreeNode leftNode : leftNodes) {
                for (TreeNode rightNode : rightNodes) {
                    TreeNode node = new TreeNode(i);
                    node.left = leftNode;
                    node.right = rightNode;

                    result.add(node);
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

        return internalValidBST(Long.MIN_VALUE, root, Long.MAX_VALUE);
    }

    private boolean internalValidBST(long minValue, TreeNode root, long maxValue) {
        if (root == null) {
            return true;
        }
        if (root.val <= minValue || root.val >= maxValue) {
            return false;
        }
        return internalValidBST(minValue, root.left, root.val) && internalValidBST(root.val, root.right, maxValue);
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
        TreeNode p = root;
        TreeNode first = null;
        TreeNode second = null;

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
                if (first != null) {
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
        return internalSortedArrayToBST(nums, 0, nums.length - 1);
    }

    private TreeNode internalSortedArrayToBST(int[] nums, int start, int end) {
        if (start == end) {
            return new TreeNode(nums[start]);
        }
        if (start > end) {
            return null;
        }
        int mid = start + (end - start) / 2;
        TreeNode root = new TreeNode(nums[mid]);
        root.left = internalSortedArrayToBST(nums, start, mid - 1);
        root.right = internalSortedArrayToBST(nums, mid + 1, end);
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
        ListNode next = slow.next;
        slow.next = null;
        TreeNode left = sortedListToBST(head);
        TreeNode right = sortedListToBST(next);
        TreeNode root = new TreeNode(slow.val);
        root.left = left;
        root.right = right;
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
        return internalBuildPreIteratorTree(0, preorder, 0, inorder.length - 1, inorder);
    }

    private TreeNode internalBuildPreIteratorTree(int preStart, int[] preorder, int inStart, int inEnd, int[] inorder) {
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
        root.left = internalBuildPreIteratorTree(preStart + 1, preorder, inStart, index - 1, inorder);
        root.right = internalBuildPreIteratorTree(preStart + index - inStart + 1, preorder, index + 1, inEnd, inorder);
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
        return internalBuildPostIteratorTree(0, inorder.length - 1, inorder, 0, postorder.length - 1, postorder);
    }

    private TreeNode internalBuildPostIteratorTree(int inStart, int inEnd, int[] inorder, int postStart, int postEnd, int[] postorder) {
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
        root.left = internalBuildPostIteratorTree(inStart, index - 1, inorder, postStart, postStart + index - inStart - 1, postorder);
        root.right = internalBuildPostIteratorTree(index + 1, inEnd, inorder, postStart + index - inStart, postEnd - 1, postorder);
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

    public TreeNode lowestCommonAncestorii(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == q || root == p) {
            return root;
        }
        if ((p.val < root.val && root.val < q.val) || (p.val > root.val && root.val > q.val)) {
            return root;
        } else if ((p.val < root.val && q.val < root.val)) {
            return lowestCommonAncestor(root.left, p, q);
        } else if (p.val > root.val && q.val > root.val) {
            return lowestCommonAncestor(root.right, p, q);
        }
        return null;
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
        if (root == p || root == q || root == null) {
            return root;
        }
        if ((p.val < root.val && root.val < q.val) || (q.val < root.val && root.val < p.val)) {
            return root;
        } else if ((p.val < root.val && q.val < root.val)) {
            return lowestCommonAncestor(root.left, p, q);
        } else if (root.val < p.val && root.val < q.val) {
            return lowestCommonAncestor(root.right, p, q);
        }
        return null;
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
        internalBinaryTreePaths(result, root, "");
        return result;
    }

    private void internalBinaryTreePaths(List<String> result, TreeNode root, String s) {
        if (root == null) {
            return;
        }
        s = s.isEmpty() ? String.valueOf(root.val) : s + "->" + root.val;
        if (root.left == null && root.right == null) {
            result.add(s);
            return;
        }
        internalBinaryTreePaths(result, root.left, s);
        internalBinaryTreePaths(result, root.right, s);
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
        internalPathSum(root, targetSum, result, new ArrayList<>());
        return result;
    }

    private void internalPathSum(TreeNode root, int targetSum, List<List<Integer>> result, List<Integer> tmp) {
        tmp.add(root.val);
        if (root.val == targetSum && root.left == null && root.right == null) {
            result.add(new ArrayList<>(tmp));
        } else {
            if (root.left != null) {
                internalPathSum(root.left, targetSum - root.val, result, tmp);
            }
            if (root.right != null) {
                internalPathSum(root.right, targetSum - root.val, result, tmp);
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

        stack.push(root);
        TreeNode prev = null;
        while (!stack.isEmpty()) {
            TreeNode node = stack.pop();

            if (node.right != null) {
                stack.push(node.right);
            }
            if (node.left != null) {
                stack.push(node.left);
            }

            if (prev != null) {
                prev.right = node;
                prev.left = null;
            }
            prev = node;
        }
    }


    private TreeNode flattenPrev = null;

    public void flattenii(TreeNode root) {
        if (root == null) {
            return;
        }
        flatten(root.right);
        flatten(root.left);
        root.right = flattenPrev;
        flattenPrev = root;
        root.left = null;
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
            Node nextLevel = current.left;
            while (current != null) {
                current.left.next = current.right;
                if (current.next != null && current.right != null) {
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
            return root;
        }
        Node current = root;
        while (current != null) {
            Node head = null;
            Node tmp = null;
            while (current != null) {
                if (current.left != null) {
                    if (head == null) {
                        head = current.left;
                        tmp = head;
                    } else {
                        tmp.next = current.left;
                        tmp = tmp.next;
                    }
                }
                if (current.right != null) {
                    if (head == null) {
                        head = current.right;
                        tmp = head;
                    } else {
                        tmp.next = current.right;
                        tmp = tmp.next;
                    }
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
        int leftValue = intervalPathSum(root.left);
        int rightValue = intervalPathSum(root.right);

        leftValue = Math.max(leftValue, 0);

        rightValue = Math.max(rightValue, 0);

        int val = leftValue + rightValue + root.val;

        maxPathSum = Math.max(maxPathSum, val);

        return Math.max(leftValue, rightValue) + root.val;
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
        return internalSumNumbers(root, 0);
    }

    private int internalSumNumbers(TreeNode root, int value) {
        if (root == null) {
            return 0;
        }
        if (root.left == null && root.right == null) {
            return value * 10 + root.val;
        }
        return internalSumNumbers(root.left, value * 10 + root.val) + internalSumNumbers(root.right, value * 10 + root.val);
    }


    // 线段树系列 //

    /**
     * 218. The Skyline Problem
     * todo
     *
     * @param buildings
     * @return
     */
    public List<List<Integer>> getSkyline(int[][] buildings) {
        return null;
    }


}
