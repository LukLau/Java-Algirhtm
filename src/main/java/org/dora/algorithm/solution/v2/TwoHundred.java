package org.dora.algorithm.solution.v2;

import org.dora.algorithm.datastructe.ListNode;
import org.dora.algorithm.datastructe.Node;
import org.dora.algorithm.datastructe.TreeNode;

import java.util.*;

/**
 * @author dora
 * @date 2019/9/3
 */
public class TwoHundred {
    /**
     * 124. Binary Tree Maximum Path Sum
     *
     * @param root
     * @return
     */
    private int MAX_PATH_SUM = Integer.MIN_VALUE;

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

    /**
     * todo 一直未熟悉
     * 109. Convert Sorted List to Binary Search Tree
     *
     * @param head
     * @return
     */
    public TreeNode sortedListToBST(ListNode head) {
        if (head == null) {
            return null;
        }
        return this.sortedListToBST(head, null);
    }

    private TreeNode sortedListToBST(ListNode head, ListNode end) {
        if (head == end) {
            return null;
        }
        ListNode fast = head;
        ListNode slow = head;
        while (fast != end && fast.next != end) {
            fast = fast.next.next;
            slow = slow.next;
        }
        TreeNode root = new TreeNode(slow.val);
        root.left = this.sortedListToBST(head, slow);
        root.right = this.sortedListToBST(slow.next, end);
        return root;
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
        int leftDepth = this.maxDepth(root.left);
        int rightDepth = this.maxDepth(root.right);
        if (Math.abs(leftDepth - rightDepth) <= 1) {
            return this.isBalanced(root.left) && this.isBalanced(root.right);
        }
        return false;
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
        if (root.left == null) {
            return 1 + this.minDepth(root.right);
        }
        if (root.right == null) {
            return 1 + this.minDepth(root.left);
        }
        int leftDepth = this.minDepth(root.left);
        int rightDepth = this.minDepth(root.right);
        return Math.min(leftDepth, rightDepth);
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
        return this.hasPathSum(root.left, sum - root.val) || this.hasPathSum(root.right, sum - root.val);
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
            return Collections.emptyList();
        }
        List<List<Integer>> ans = new ArrayList<>();
        this.pathSum(ans, new ArrayList<Integer>(), root, sum);
        return ans;

    }

    private void pathSum(List<List<Integer>> ans, List<Integer> integers, TreeNode root, int sum) {
        if (root == null) {
            return;
        }
        integers.add(root.val);

        if (root.left == null && root.right == null && root.val == sum) {
            ans.add(new ArrayList<>(integers));
        } else {
            if (root.left != null) {
                this.pathSum(ans, integers, root.left, sum - root.val);
            }
            if (root.right != null) {
                this.pathSum(ans, integers, root.right, sum - root.val);
            }
        }
        integers.remove(integers.size() - 1);

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
            TreeNode node = stack.pop();
            if (node.right != null) {
                stack.push(node.right);
            }
            if (node.left != null) {
                stack.push(node.left);
            }
            if (prev != null) {
                prev.left = null;
                prev.right = node;
            }
            prev = node;
        }
    }

    /**
     * todo 不懂之处
     * 115. Distinct Subsequences
     *
     * @param s
     * @param t
     * @return
     */
    public int numDistinct(String s, String t) {
        if (s == null || t == null) {
            return -1;
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
     * todo 不懂之处
     * 116. Populating Next Right Pointers in Each Node
     *
     * @param root
     * @return
     */
    public Node connect(Node root) {
        if (root == null) {
            return null;
        }
        Node head = root;
        while (head.left != null) {
            Node next = head.left;

            while (next != null) {
                next.next = head.right;
                head = head.next;
            }

            head = next;
        }
        return null;
    }

    /**
     * 118. Pascal's Triangle
     *
     * @param numRows
     * @return
     */
    public List<List<Integer>> generate(int numRows) {
        if (numRows <= 0) {
            return Collections.emptyList();
        }
        List<List<Integer>> ans = new ArrayList<>();
        for (int i = 0; i < numRows; i++) {

            List<Integer> tmp = new ArrayList<>();

            tmp.add(1);
            for (int j = 1; j <= i - 1; j++) {
                int value = ans.get(i - 1).get(j - 1) + ans.get(i - 1).get(j);
                tmp.add(value);
            }

            if (i > 0) {
                tmp.add(i);
            }
            ans.add(tmp);
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
            return Collections.emptyList();
        }
        List<Integer> ans = new ArrayList<>();

        ans.add(1);
        for (int i = 0; i <= rowIndex; i++) {


            for (int j = i - 1; j >= 1; j--) {
                int value = ans.get(j) + ans.get(j - 1);

                ans.set(j, value);
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
        if (triangle == null || triangle.isEmpty()) {
            return 0;
        }
        int size = triangle.size();
        List<Integer> ans = triangle.get(size - 1);

        for (int i = size - 2; i >= 0; i--) {

            for (int j = 0; j < triangle.get(i).size(); j++) {

                int value = Math.min(ans.get(j), ans.get(j + 1)) + triangle.get(i).get(j);

                ans.set(j, value);
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
        int result = 0;
        int min = prices[0];
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > min) {
                result = Math.max(result, prices[i] - min);
            }
            prices[i] = min;

        }
        return result;

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

        int min = Integer.MAX_VALUE;
        for (int i = 0; i < prices.length; i++) {
            if (prices[i] < min) {
                min = prices[i];
            } else {
                result += prices[i] - min;
                min = prices[i];
            }
        }
        return result;
    }

    /**
     * todo 比较难需要两边遍历
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

        int[] leftProfit = new int[n];

        int leftValue = prices[0];

        int leftResult = 0;

        for (int i = 1; i < n; i++) {
            if (prices[i] < leftValue) {
                leftValue = prices[i];
            }

            leftResult = Math.max(leftResult, prices[i] - leftValue);

            leftProfit[i] = leftResult;


        }

        int[] rightProfit = new int[n + 1];

        int rightResult = 0;

        int rightValue = prices[n - 1];

        for (int i = n - 2; i >= 0; i--) {
            if (prices[i] > rightValue) {
                rightValue = prices[i];
            }
            rightResult = Math.max(rightResult, rightValue - prices[i]);

            rightProfit[i] = rightResult;
        }
        int result = 0;

        for (int i = 0; i < n; i++) {
            result = Math.max(result, leftProfit[i] + rightProfit[i + 1]);
        }
        return result;
    }

    public int maxPathSum(TreeNode root) {
        if (root == null) {
            return 0;
        }
        this.dfs(root);
        return MAX_PATH_SUM;

    }

    private int dfs(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int leftValue = this.dfs(root.left);
        int rightValue = this.dfs(root.right);

        leftValue = Math.max(leftValue, 0);

        rightValue = Math.max(rightValue, 0);

        int value = leftValue + rightValue + root.val;

        MAX_PATH_SUM = Math.max(MAX_PATH_SUM, value);

        return Math.max(leftValue, rightValue) + root.val;
    }

    /**
     * 125. Valid Palindrome
     *
     * @param s
     * @return
     */
    public boolean isPalindrome(String s) {
        if (s == null || s.isEmpty()) {
            return true;
        }
        int left = 0;

        int right = s.length() - 1;

        while (left < right) {
            if (!Character.isLetterOrDigit(s.charAt(left))) {
                left++;
            } else if (!Character.isLetterOrDigit(s.charAt(right))) {
                right--;
            } else {
                if (Character.toLowerCase(s.charAt(left)) == Character.toLowerCase(s.charAt(right))) {
                    left++;
                    right--;
                } else {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * 126. Word Ladder IIt
     * todo 太难
     *
     * @param beginWord
     * @param endWord
     * @param wordList
     * @return
     */
    public List<List<String>> findLadders(String beginWord, String endWord, List<String> wordList) {
        return null;

    }

    /**
     * 127. Word Ladder
     *
     * @param beginWord
     * @param endWord
     * @param wordList
     * @return
     */
    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        return 0;
    }

    /**
     * 128. Longest Consecutive Sequence
     *
     * @param nums
     * @return
     */
    public int longestConsecutive(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        HashMap<Integer, Integer> hashMap = new HashMap<>();
        int result = 0;
        for (int num : nums) {
            if (hashMap.containsKey(num)) {
                continue;
            }
            int leftValue = hashMap.getOrDefault(num - 1, 0);
            int rightValue = hashMap.getOrDefault(num + 1, 0);

            int sum = leftValue + rightValue + 1;

            result = Math.max(result, sum);

            hashMap.put(num - leftValue, sum);

            hashMap.put(num + rightValue, sum);

            hashMap.put(num, sum);
        }
        return result;
    }


    /**
     * 129. Sum Root to Leaf Numbers
     *
     * @param root
     * @return
     */
    public int sumNumbers(TreeNode root) {
        if (root == null) {
            return 0;
        }
        if (root.left == null && root.right == null) {
            return root.val;
        }
        return this.sumNumbers(root.left, root.val) + this.sumNumbers(root.right, root.val);
    }

    private int sumNumbers(TreeNode root, int sum) {
        if (root == null) {
            return 0;
        }
        if (root.left == null && root.right == null) {
            return sum * 10 + root.val;
        }
        return this.sumNumbers(root.left, sum * 10 + root.val) + this.sumNumbers(root.right, sum * 10 + root.val);

    }

    /**
     * 130. Surrounded Regions
     *
     * @param board
     */
    public void solve(char[][] board) {
        if (board == null || board.length == 0) {
            return;
        }
        int row = board.length;

        int column = board[0].length;

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                boolean match = (i == 0 || i == row - 1 || j == 0 || j == column - 1);
                if (match && board[i][j] == 'O') {
                    this.solveMatrix(i, j, board);
                }
            }
        }

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (board[i][j] == '0') {
                    board[i][j] = 'O';
                } else if (board[i][j] == 'O') {
                    board[i][j] = 'X';
                }
            }
        }

        return;
    }

    private void solveMatrix(int row, int column, char[][] board) {
        if (row < 0 || row >= board.length || column < 0 || column >= board[0].length || board[row][column] != 'O') {
            return;
        }
        board[row][column] = '0';
        this.solveMatrix(row - 1, column, board);
        this.solveMatrix(row + 1, column, board);
        this.solveMatrix(row, column - 1, board);
        this.solveMatrix(row, column + 1, board);

    }

    /**
     * todo  完全不懂
     * 131. Palindrome Partitioning
     *
     * @param s
     * @return
     */
    public List<List<String>> partition(String s) {
        if (s == null || s.isEmpty()) {
            return Collections.emptyList();
        }
        List<List<String>> ans = new ArrayList<>();
        for (int i = 0; i < s.length(); i++) {

        }
        return null;

    }

    /**
     * todo 未懂思路
     * 132. Palindrome Partitioning II
     *
     * @param s
     * @return
     */
    public int minCut(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        }
        int n = s.length();

        boolean[][] dp = new boolean[n + 1][n + 1];

        int[] cut = new int[n];
        for (int i = 0; i < n; i++) {
            int min = i;
            for (int j = 0; j <= i; j++) {
                if (s.charAt(j) == s.charAt(i) && (i - j < 2 || dp[j + 1][i - 1])) {

                    dp[j][i] = true;

                    min = j == 0 ? 0 : Math.min(1 + cut[j - 1], min);

                }
            }
            cut[i] = min;
        }
        return cut[n - 1];
    }


    /**
     * 134. Gas Station
     *
     * @param gas
     * @param cost
     * @return
     */
    public int canCompleteCircuit(int[] gas, int[] cost) {
        if (gas == null || cost == null) {
            return 0;
        }
        int result = 0;

        int local = 0;

        int current = 0;

        for (int i = 0; i < gas.length; i++) {
            local += gas[i] - cost[i];
            result += gas[i] - cost[i];
            if (local < 0) {
                local = 0;
                current = i + 1;
            }
        }
        return result < 0 ? -1 : current;

    }

    /**
     * 135. Candy
     *
     * @param ratings
     * @return
     */
    public int candy(int[] ratings) {
        if (ratings == null || ratings.length == 0) {
            return 0;
        }
        int n = ratings.length;
        int[] left = new int[n];

        for (int i = 0; i < n; i++) {
            left[i] = 1;
        }

        for (int i = 1; i < n; i++) {

            if (ratings[i] > ratings[i - 1] && left[i] < left[i - 1] + 1) {
                left[i] = left[i - 1] + 1;
            }
        }
        int[] right = new int[n];
        for (int i = 0; i < n; i++) {
            right[i] = 1;
        }
        for (int i = n - 2; i >= 0; i--) {
            if (ratings[i] > ratings[i + 1] && right[i] < right[i + 1] + 1) {
                right[i] = right[i + 1] + 1;
            }
        }

        int result = 0;

        for (int i = 0; i < n; i++) {
            result += Math.max(left[i], right[i]);
        }
        return result;
    }


    /**
     * 136. Single Number
     *
     * @param nums
     * @return
     */
    public int singleNumber(int[] nums) {
        if (nums == null || nums.length == 0) {
            return -1;
        }
        int result = 0;
        for (int num : nums) {
            result ^= num;
        }
        return result;
    }


    /**
     * 137. Single Number II
     *
     * @param nums
     * @return
     */
    public int singleNumberII(int[] nums) {
        return 0;
    }


    /**
     * 138. Copy List with Random Pointer
     *
     * @param head
     * @return
     */
    public Node copyRandomList(Node head) {
        if (head == null) {
            return null;
        }
        Node current = head;
        while (current != null) {

            Node node = new Node(current.val, current.next, null);

            current.next = node;

            current = node.next;

        }
        current = head;

        while (current != null) {

            Node next = current.next;

            if (current.random != null) {
                next.random = current.random.next;
            }
            current = next.next;
        }
        current = head;

        Node pHead = current.next;

        while (current.next != null) {
            Node tmp = current.next;

            current.next = tmp.next;

            current = tmp;
        }
        return pHead;
    }

    /**
     * 139. Word Break
     *
     * @param s
     * @param wordDict
     * @return
     */
    public boolean wordBreak(String s, List<String> wordDict) {
        if (s == null || wordDict == null) {
            return false;
        }
        return false;
    }


    /**
     * 140. Word Break II
     *
     * @param s
     * @param wordDict
     * @return
     */
    public List<String> wordBreakII(String s, List<String> wordDict) {
        return null;
    }

    /**
     * 141. Linked List Cycle
     *
     * @param head
     * @return
     */
    public boolean hasCycle(ListNode head) {
        if (head == null || head.next == null) {
            return false;
        }
        ListNode slow = head;

        ListNode fast = head;
        while (fast.next != null && fast.next.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast) {
                return true;
            }
        }
        return false;
    }


}
