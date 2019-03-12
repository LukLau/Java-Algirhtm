package org.dora.algorithm.solution;

import org.dora.algorithm.datastructe.ListNode;
import org.dora.algorithm.datastructe.Node;
import org.dora.algorithm.datastructe.Point;
import org.dora.algorithm.datastructe.TreeNode;

import java.util.*;

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
    private static int ans = 0;

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

    /**
     * @param s
     * @return
     */
    public boolean isPalindrome(String s) {
        if (s == null || s.length() == 0) {
            return true;
        }
        int left = 0;
        int right = s.length() - 1;
        while (left < right) {
            if (!Character.isLetterOrDigit(s.charAt(left))) {
                left++;
                continue;
            }
            if (!Character.isLetterOrDigit(s.charAt(right))) {
                right--;
                continue;
            }
            if (Character.toLowerCase(s.charAt(left)) == Character.toLowerCase(s.charAt(right))) {
                left++;
                right--;
            } else {
                return false;
            }
        }
        return true;
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
        for (int i = 0; i < nums.length; i++) {
            if (!hashMap.containsKey(nums[i])) {
                int left = hashMap.getOrDefault(nums[i] - 1, 0);
                int right = hashMap.getOrDefault(nums[i] + 1, 0);

                int value = left + right + 1;
                result = Math.max(result, value);

                hashMap.put(nums[i], value);
                hashMap.put(nums[i] - left, value);
                hashMap.put(nums[i] + right, value);
            }
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
        return sumNumbers(root.left, root.val) + sumNumbers(root.right, root.val);
    }

    private int sumNumbers(TreeNode root, int value) {
        if (root == null) {
            return 0;
        }
        if (root.left == null && root.right == null) {
            return value * 10 + root.val;
        }
        return sumNumbers(root.left, value * 10 + root.val) + sumNumbers(root.right, value * 10 + root.val);
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
        int[][] matrix = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        Queue<Point> queue = new LinkedList<>();
        for (int i = 0; i < board.length; i++) {

            for (int j = 0; j < board[i].length; j++) {
                boolean edge = i == 0 || i == board.length - 1 || j == 0 || j == board[i].length - 1;
                if (edge && board[i][j] == 'O') {
                    Point point = new Point(i, j);
                    queue.offer(point);
                    board[i][j] = 'B';
                    while (!queue.isEmpty()) {
                        Point node = queue.poll();
                        for (int k = 0; k < 4; k++) {
                            int x = matrix[k][0] + node.x;
                            int y = matrix[k][1] + node.y;
                            if (x >= 0 && x <= board.length - 1 && y >= 0 && y <= board[i].length - 1 && board[x][y] == 'O') {
                                board[x][y] = 'B';
                                queue.offer(new Point(x, y));
                            }
                        }
                    }
                }
            }
        }
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length; j++) {
                if (board[i][j] == 'B') {
                    board[i][j] = 'O';
                } else if (board[i][j] == 'O') {
                    board[i][j] = 'X';
                }
            }
        }
    }

    private void solve(int i, int j, char[][] board) {
        if (i > 0 && board[i - 1][j] == 'O') {
            board[i - 1][j] = 'B';
            solve(i - 1, j, board);
        }
        if (i < board.length - 1 && board[i + 1][j] == 'O') {
            board[i + 1][j] = 'B';
            solve(i + 1, j, board);
        }
        if (j > 0 && board[i][j - 1] == 'O') {
            board[i][j - 1] = 'B';
            solve(i, j - 1, board);
        }
        if (j < board[i].length - 1 && board[i][j + 1] == 'O') {
            board[i][j + 1] = 'B';
            solve(i, j + 1, board);
        }
    }

    /**
     * 131. Palindrome Partitioning
     *
     * @param s
     * @return
     */
    public List<List<String>> partition(String s) {
        if (s == null || s.length() == 0) {
            return new ArrayList<>();
        }
        List<List<String>> ans = new ArrayList<>();
        dfs(ans, new ArrayList<>(), 0, s);
        return ans;
    }

    private void dfs(List<List<String>> ans, List<String> tmp, int left, String s) {
        if (left == s.length()) {
            ans.add(new ArrayList<>(tmp));
        }
        for (int i = left; i < s.length(); i++) {
            if (isValid(s, left, i)) {
                tmp.add(s.substring(left, i + 1));
                dfs(ans, tmp, i + 1, s);
                tmp.remove(tmp.size() - 1);
            }
        }
    }

    private boolean isValid(String s, int left, int right) {
        if (left > right) {
            return false;
        }
        while (left < right) {
            if (s.charAt(left) == s.charAt(right)) {
                left++;
                right--;
            } else {
                return false;
            }
        }
        return true;
    }

    /**
     * 132. Palindrome Partitioning II
     *
     * @param s
     * @return
     */
    public int minCut(String s) {
        if (s == null || s.length() == 0) {
            return 0;
        }
        int n = s.length();
        int[] curr = new int[n];
        boolean[][] dp = new boolean[n][n];
        for (int i = 0; i < n; i++) {
            int min = i;
            for (int j = 0; j <= i; j++) {
                if (s.charAt(j) == s.charAt(i) && (i - j < 2 || dp[j + 1][i - 1])) {
                    dp[j][i] = true;
                    min = j == 0 ? 0 : Math.min(min, curr[j - 1] + 1);
                }
            }
            curr[i] = min;
        }
        return curr[n - 1];
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
        int sum = 0;
        int total = 0;
        int index = 0;
        for (int i = 0; i < gas.length; i++) {
            total += gas[i] - cost[i];
            sum += gas[i] - cost[i];
            if (sum < 0) {
                sum = 0;
                index = i + 1;
            }
        }
        return total < 0 ? -1 : index;
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
        int[] candy = new int[n];
        for (int i = 0; i < n; i++) {
            candy[i] = 1;
        }
        for (int i = 1; i < n; i++) {
            if (ratings[i] > ratings[i - 1]) {
                candy[i] = Math.max(candy[i], candy[i - 1] + 1);
            }
        }
        for (int i = n - 2; i >= 0; i--) {
            if (ratings[i] > ratings[i - 1]) {
                candy[i] = Math.max(candy[i], candy[i - 1] + 1);
            }
        }
        int result = 0;
        for (int candys : candy) {
            result += candys;
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
     * 138. Copy List with Random Pointer
     *
     * @param head
     * @return
     */
    public Node copyRandomList(Node head) {
        if (head == null) {
            return null;
        }
        Node currNode = head;

        while (currNode != null) {

            Node tmp = new Node(currNode.val, currNode.next, currNode.random);

            currNode.next = tmp;

            currNode = tmp.next;

        }
        currNode = head;
        while (currNode != null) {
            Node node = currNode.next;
            if (currNode.random != null) {
                node.random = currNode.random.next;
            }
            currNode = node.next;
        }
        currNode = head;
        Node copyNode = currNode.next;
        while (currNode.next != null) {
            Node tmp = currNode.next;

            currNode.next = tmp.next;

            currNode = tmp;
        }
        return copyNode;
    }


}
