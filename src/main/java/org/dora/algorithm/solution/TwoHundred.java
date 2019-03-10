package org.dora.algorithm.solution;

import org.dora.algorithm.datastructe.ListNode;
import org.dora.algorithm.datastructe.TreeNode;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Stack;

/**
 * @author lauluk
 * @date 2019/03/07
 */
public class TwoHundred {
    /**
     * 124. Binary Tree Maximum Path Sum
     *
     * @param root
     * @return
     */
    private static int maxPathSum = Integer.MIN_VALUE;

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
            LinkedList<Integer> tmp = new LinkedList<>();
            for (int i = 0; i < size; i++) {
                TreeNode node = list.pollFirst();
                if (!isLeft) {
                    tmp.addLast(node.val);
                } else {
                    tmp.addFirst(node.val);
                }
                if (node.left != null) {
                    list.addLast(node.left);
                }
                if (node.right != null) {
                    list.addLast(node.right);
                }
            }
            isLeft = !isLeft;
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
        return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
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
        if (preorder.length == 0 || inorder.length == 0) {
            return null;
        }
        return buildTree(0, 0, inorder.length - 1, preorder, inorder);
    }

    private TreeNode buildTree(int preStart, int inStart, int inEnd, int[] preorder, int[] inorder) {
        if (preStart > preorder.length || inStart > inEnd) {
            return null;
        }
        TreeNode root = new TreeNode(preorder[preStart]);
        int idx = 0;
        for (int i = inStart; i <= inEnd; i++) {
            if (inorder[i] == root.val) {
                idx = i;
                break;
            }
        }
        root.left = buildTree(preStart + 1, inStart, idx - 1, preorder, inorder);
        root.right = buildTree(preStart + idx - inStart + 1, idx + 1, inEnd, preorder, inorder);
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
        return buildTree(0, inorder.length - 1, inorder, 0, postorder.length - 1, postorder);
    }

    private TreeNode buildTree(int inStart, int inEnd, int[] inorder, int postStart, int postEnd, int[] postorder) {
        if (inStart > inEnd) {
            return null;
        }
        if (postStart > postEnd) {
            return null;
        }
        TreeNode root = new TreeNode(postorder[postEnd]);
        int idx = 0;
        for (int i = inStart; i <= inEnd; i++) {
            if (inorder[i] == root.val) {
                idx = i;
                break;
            }
        }
        root.left = buildTree(inStart, idx - 1, inorder, postStart, postStart + idx - inStart - 1, postorder);
        root.right = buildTree(idx + 1, inEnd, inorder, postStart + idx - inStart, postEnd - 1, postorder);
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
            return new ArrayList<>();
        }
        LinkedList<List<Integer>> ans = new LinkedList<>();
        LinkedList<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            List<Integer> tmp = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                TreeNode node = queue.pollFirst();
                if (node.left != null) {
                    queue.addLast(node.left);
                }
                if (node.right != null) {
                    queue.addLast(node.right);
                }
                tmp.add(node.val);
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
        return sortedArrayToBST(nums, 0, nums.length - 1);

    }

    private TreeNode sortedArrayToBST(int[] nums, int start, int end) {
        if (start > end) {
            return null;
        }
        int mid = start + (end - start) / 2;
        TreeNode root = new TreeNode(nums[mid]);
        root.left = sortedArrayToBST(nums, start, mid - 1);
        root.right = sortedArrayToBST(nums, mid + 1, end);
        return root;
    }

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
        return toBST(head, null);
    }

    private TreeNode toBST(ListNode head, ListNode tail) {
        if (head == tail) {
            return null;

        }
        ListNode fast = head;
        ListNode slow = head;

        while (fast != tail && fast.next != tail) {
            fast = fast.next.next;
            slow = slow.next;
        }
        TreeNode thead = new TreeNode(slow.val);
        thead.left = toBST(head, slow);
        thead.right = toBST(slow.next, tail);
        return thead;
    }

    /**
     * 110. Balanced Binary Tree
     *
     * @param root
     * @return
     */
    public boolean isBalanced(TreeNode root) {
        if (root == null) {
            return true;
        }
        return dfs(root) != -1;
    }

    private int dfs(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int left = dfs(root.left);
        if (left == -1) {
            return -1;
        }
        int right = dfs(root.right);
        if (right == -1) {
            return -1;
        }
        if (Math.abs(left - right) > 1) {
            return -1;
        }
        return 1 + Math.max(left, right);
    }

    /**
     * 111. Minimum Depth of Binary Tree
     *
     * @param root
     * @return
     */
    public int minDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        if (root.left == null && root.right == null) {
            return 1;
        }
        if (root.left == null) {
            return 1 + minDepth(root.right);
        }
        if (root.right == null) {
            return 1 + minDepth(root.left);
        }
        return 1 + Math.min(minDepth(root.left), minDepth(root.right));
    }

    /**
     * 112. Path Sum
     *
     * @param root
     * @param sum
     * @return
     */
    public boolean hasPathSum(TreeNode root, int sum) {
        if (root == null) {
            return false;
        }
        if (root.left == null && root.right == null && root.val == sum) {
            return true;
        }
        return hasPathSum(root.left, sum - root.val) || hasPathSum(root.right, sum - root.val);
    }

    private boolean hasPathSumDfs(TreeNode root, int value) {
        if (root == null) {
            return false;
        }
        if (value == 0) {
            return false;
        }
        if (root.left == null && root.right == null && root.val == value) {
            return true;
        }
        return hasPathSumDfs(root.left, value - root.val) || hasPathSumDfs(root.right, value - root.val);
    }

    /**
     * 113. Path Sum II
     *
     * @param root
     * @param sum
     * @return
     */
    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        if (root == null) {
            return new ArrayList<>();
        }
        List<List<Integer>> ans = new ArrayList<>();
        pathSum(ans, new ArrayList<>(), root, sum);
        return ans;
    }

    private void pathSum(List<List<Integer>> ans, List<Integer> tmp, TreeNode root, int sum) {
        if (root == null) {
            return;
        }
        tmp.add(root.val);
        if (root.left == null && root.right == null && root.val == sum) {
            ans.add(new ArrayList<>(tmp));
            tmp.remove(tmp.size() - 1);
            return;
        }
        if (root.left != null) {
            pathSum(ans, tmp, root.left, sum - root.val);
        }
        if (root.right != null) {
            pathSum(ans, tmp, root.right, sum - root.val);
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
        TreeNode prev = null;
        stack.push(root);
        while (!stack.isEmpty()) {
            TreeNode node = stack.pop();
            if (node.right != null) {
                stack.push(node.right);
            }
            if (node.left != null) {
                stack.push(node.left);
            }
            if (prev == null) {
                prev = node;
            } else {
                prev.left = null;
                prev.right = node;
                prev = prev.right;
            }
        }
    }

    /**
     * 115. Distinct Subsequences
     *
     * @param s
     * @param t
     * @return
     */
    public int numDistinct(String s, String t) {
        if (s == null || t == null) {
            return 0;
        }
        if (s.equals(t)) {
            return 0;
        }
        int m = s.length();
        int n = t.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 0; i <= m; i++) {
            dp[i][0] = 1;
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                dp[i][j] = (s.charAt(i - 1) == t.charAt(j - 1) ? dp[i - 1][j - 1] : 0) + dp[i - 1][j];
            }
        }
        return dp[m][n];
    }

    /**
     * 118. Pascal's Triangle
     *
     * @param numRows
     * @return
     */
    public List<List<Integer>> generate(int numRows) {
        if (numRows <= 0) {
            return new ArrayList<>();
        }
        List<List<Integer>> ans = new ArrayList<>();

        for (int i = 0; i <= numRows - 1; i++) {
            List<Integer> tmp = new ArrayList<>();

            tmp.add(1);

            for (int j = 1; j < i; j++) {

                int sum = ans.get(i - 1).get(j - 1) + ans.get(i - 1).get(j);

                tmp.add(sum);
            }
            if (i > 0) {
                tmp.add(1);
            }
            ans.add(new ArrayList<>(tmp));
        }
        return ans;
    }

    /**
     * 119. Pascal's Triangle II
     *
     * @param rowIndex
     * @return
     */
    public List<Integer> getRow(int rowIndex) {
        if (rowIndex < 0) {
            return new ArrayList<>();
        }
        List<Integer> ans = new ArrayList<>();


        ans.add(1);

        for (int i = 0; i <= rowIndex; i++) {

            for (int j = i - 1; j >= 1; j--) {
                ans.set(j, ans.get(j) + ans.get(j - 1));
            }
            if (i > 0) {
                ans.add(1);
            }
        }
        return ans;
    }

    /**
     * 120. Triangle
     *
     * @param triangle
     * @return
     */
    public int minimumTotal(List<List<Integer>> triangle) {
        if (triangle.isEmpty()) {
            return 0;
        }
        int size = triangle.size();

        List<Integer> ans = triangle.get(size - 1);

        for (int i = size - 2; i >= 0; i--) {

            for (int j = 0; j < triangle.get(i).size(); j++) {


                ans.set(j, triangle.get(i).get(j) + Math.min(ans.get(j), ans.get(j + 1)));
            }
        }
        return ans.get(0);
    }

    /**
     * 121. Best Time to Buy and Sell Stock
     *
     * @param prices
     * @return
     */
    public int maxProfit(int[] prices) {
        if (prices == null || prices.length == 0) {
            return 0;
        }
        int maxProfit = 0;
        int min = prices[0];
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] < min) {
                min = prices[i];
            }
            if (prices[i] > min) {
                maxProfit = Math.max(maxProfit, prices[i] - min);
            }
        }
        return maxProfit;
    }

    /**
     * 122. Best Time to Buy and Sell Stock II
     *
     * @param prices
     * @return
     */
    public int maxProfitII(int[] prices) {
        if (prices == null || prices.length == 0) {
            return 0;
        }
        int result = 0;
        int min = prices[0];
        for (int price : prices) {
            if (price > min) {
                result += price - min;
            }
            min = price;
        }
        return result;
    }

    /**
     * 123. Best Time to Buy and Sell Stock III
     *
     * @param prices
     * @return
     */
    public int maxProfitIII(int[] prices) {
        if (prices == null || prices.length == 0) {
            return 0;
        }
        int n = prices.length;
        int[] left = new int[n];
        int minPrices = prices[0];
        int maxLeft = 0;
        for (int i = 1; i < n; i++) {
            if (prices[i] < minPrices) {
                minPrices = prices[i];
            }
            maxLeft = Math.max(maxLeft, prices[i] - minPrices);
            left[i] = maxLeft;
        }
        int[] right = new int[n + 1];

        int maxPrices = prices[n - 1];

        int maxRight = 0;
        for (int i = n - 2; i >= 0; i--) {
            if (prices[i] > maxPrices) {
                maxPrices = prices[i];
            }
            maxRight = Math.max(maxRight, maxPrices - prices[i]);
            right[i] = maxRight;
        }

        int result = 0;

        for (int i = 0; i < n; i++) {
            result = Math.max(result, left[i] + right[i + 1]);
        }
        return result;

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
        dfsMaxPathSum(root);
        return maxPathSum;
    }

    private int dfsMaxPathSum(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int left = dfsMaxPathSum(root.left);
        int right = dfsMaxPathSum(root.right);
        maxPathSum = Math.max(maxPathSum, left + right + root.val);
        return Math.max(left, right) + root.val;
    }

}
