package org.dora.algorithm.leetcode;

import org.dora.algorithm.datastructe.ListNode;
import org.dora.algorithm.datastructe.Node;
import org.dora.algorithm.datastructe.TreeNode;

import java.util.*;

/**
 * @author dora
 * @date 2019-04-29
 */
public class SecondPage {

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
     * 106. Construct Binary Tree from Inorder and Postorder Traversal
     *
     * @param inorder
     * @param postorder
     * @return
     */
    public TreeNode buildTree(int[] inorder, int[] postorder) {

        if (inorder == null || postorder == null) {
            return null;
        }
        return this.buildPostTree(0, inorder.length - 1, inorder, 0, postorder.length - 1, postorder);
    }

    private TreeNode buildPostTree(int inStart, int inEnd, int[] inorder, int postStart, int postEnd, int[] postorder) {
        if (inStart > inEnd || postStart > postEnd) {
            return null;
        }
        TreeNode root = new TreeNode(postorder[postEnd]);
        int index = 0;
        for (int i = 0; i < inorder.length; i++) {
            if (inorder[i] == root.val) {
                index = i;
                break;
            }

        }
        root.left = this.buildPostTree(inStart, index - 1, inorder, postStart, postStart + index - inStart - 1, postorder);

        root.right = this.buildPostTree(index + 1, inEnd, inorder, postStart + index - inStart, postEnd - 1, postorder);
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

        LinkedList<TreeNode> deque = new LinkedList<>();

        deque.add(root);

        while (!deque.isEmpty()) {

            List<Integer> tmp = new ArrayList<>();
            int size = deque.size();

            for (int i = 0; i < size; i++) {

                TreeNode node = deque.pop();

                if (node.left != null) {
                    deque.add(node.left);
                }
                if (node.right != null) {
                    deque.add(node.right);
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
        return this.buildBST(0, nums.length - 1, nums);
    }

    private TreeNode buildBST(int start, int end, int[] nums) {
        if (start > end) {
            return null;
        }
        int mid = start + (end - start) / 2;
        TreeNode root = new TreeNode(nums[mid]);
        root.left = this.buildBST(start, mid - 1, nums);
        root.right = this.buildBST(mid + 1, end, nums);
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
        return this.buildBSTByList(head, null);
    }

    private TreeNode buildBSTByList(ListNode start, ListNode end) {

        if (start == end) {
            return null;
        }


        ListNode fast = start;
        ListNode slow = start;

        while (fast != end && fast.next != end) {
            fast = fast.next.next;
            slow = slow.next;
        }
        TreeNode root = new TreeNode(slow.val);

        root.left = this.buildBSTByList(start, slow);

        root.right = this.buildBSTByList(slow.next, end);

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
        if (Math.abs(this.maxDepth(root.left) - this.maxDepth(root.right)) <= 1) {
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
        return 1 + Math.min(this.minDepth(root.left), this.minDepth(root.right));
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
            return new ArrayList<>();
        }
        List<List<Integer>> ans = new ArrayList<>();
        this.pathSum(ans, new ArrayList<>(), root, sum);
        return ans;
    }

    private <E> void pathSum(List<List<Integer>> ans, List<Integer> tmp, TreeNode root, int sum) {
        tmp.add(root.val);
        if (root.left == null && root.right == null && root.val == sum) {
            ans.add(new ArrayList<>(tmp));
            tmp.remove(tmp.size() - 1);
            return;
        } else {
            if (root.left != null) {
                this.pathSum(ans, tmp, root.left, sum - root.val);
            }
            if (root.right != null) {
                this.pathSum(ans, tmp, root.right, sum - root.val);
            }
        }
        tmp.remove(tmp.size() - 1);
    }


    /**
     * 114. Flatten Binary Tree to Linked Lis
     *
     * @param root
     */
    public void flatten(TreeNode root) {
        if (root == null) {
            return;
        }
        Stack<TreeNode> stack = new Stack<>();
        TreeNode prev = null;
        TreeNode node = root;
        stack.push(node);
        while (!stack.isEmpty()) {
            node = stack.pop();
            if (node.right != null) {
                stack.push(node.right);
            }
            if (node.left != null) {
                stack.push(node.left);
            }
            if (prev != null) {
                prev.right = node;
            }
            prev = node;
            node.left = null;
        }
        return;
    }

    /**
     * 115. Distinct Subsequences
     * trick: 生成最小无差别 子序列 步数
     * 由于子序列 故无需考虑 下标连续
     * 同时如果 s[i-1] = t[j] then s[i] = t[j]
     *
     * @param s
     * @param t
     * @return
     */
    public int numDistinct(String s, String t) {
        if (s == null || t == null) {
            return 0;
        }
        int m = s.length();
        int n = t.length();
        int[][] dp = new int[m + 1][n + 1];
        // 空集 是任何序列的子集
        // 判断一个字符串 在序列子集数量时
        // 空集 有且只有一个
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
     * 116. Populating Next Right Pointers in Each Node
     *
     * @param root
     * @return
     */
    public Node connect(Node root) {
        if (root == null) {
            return null;
        }
        Deque<Node> deque = new LinkedList<>();
        Node node = root;
        deque.add(node);
        while (!deque.isEmpty()) {
            int size = deque.size();
            Node prev = null;
            for (int i = 0; i < size; i++) {
                node = deque.poll();
                if (prev == null) {
                    prev = node;
                } else {
                    prev.next = null;
                }
            }
        }
        return null;
    }

    /**
     * 118. Pascal's Triangle
     * <p>
     * 动态规划问题
     * dp[i][j] = 第i行第j列数值应该为
     * dp[row][column] = dp[row-1][column-1] + dp[row-1][column]
     *
     * @param numRows
     * @return
     */
    public List<List<Integer>> generate(int numRows) {
        if (numRows <= 0) {
            return new ArrayList<>();
        }
        List<List<Integer>> ans = new ArrayList<>();


        List<Integer> tmp = new ArrayList<>();


        for (int i = 0; i <= numRows - 1; i++) {


            tmp.add(0, 1);

            for (int j = 1; j < i; j++) {

                int value = tmp.get(j) + tmp.get(j + 1);


                tmp.set(j, value);

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

                int value = ans.get(j - 1) + ans.get(j);

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
     * dp[row][column] = dp[row+1][column] + dp[row+1][column+1]
     *
     * @param
     * @return
     */
    public int minimumTotal(List<List<Integer>> triangle) {
        if (triangle == null || triangle.isEmpty()) {
            return 0;
        }
        int size = triangle.size();

        List<Integer> dp = triangle.get(size - 1);

        for (int i = size - 2; i >= 0; i--) {
            for (int j = 0; j < triangle.get(i).size(); j++) {

                dp.set(j, Math.min(dp.get(j), dp.get(j + 1)) + triangle.get(i).get(j));
            }
        }

        return dp.get(0);
    }


    /**
     * 121. Best Time to Buy and Sell Stock
     * <p>
     * 由于只允许卖一次 遍历 股票表
     * 当股票值比 最小值小的时候 更新最小值
     * 比最小值大的时候 就可以卖出股票
     * 获取全局最大的 价格差
     * 全局 卖一次的最大值 即 只卖一次 获取利益最大
     * </p>
     *
     * @param prices
     * @return
     */
    public int maxProfit(int[] prices) {
        if (prices == null || prices.length == 0) {
            return 0;
        }
        int minPrices = prices[0];
        int result = 0;
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] < minPrices) {
                minPrices = prices[i];
            } else {
                result = Math.max(result, prices[i] - minPrices);
            }
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
        int minPrice = prices[0];
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] < minPrice) {
                minPrice = prices[i];
            } else {
                result += prices[i] - minPrice;
                minPrice = prices[i];
            }
        }
        return result;
    }


    /**
     * 123. Best Time to Buy and Sell Stock III
     * <p>
     * trick: 动态规划
     *
     * @param prices
     * @return
     */
    public int maxProfitIII(int[] prices) {
        if (prices == null || prices.length == 0) {
            return 0;
        }

        /**
         * 总共交易两次
         * 再卖出第二次之前 必须进行第一次交易
         * 最大利润 = left(i) + right(i+1)
         * 第i天以及i天之前 进行一次交易的最大值
         * 第 i + 1 天之后 第二次交易的最大值
         * 获取利润的全局值
         *
         */
        int length = prices.length;

        int[] left = new int[length];


        int minPrice = prices[0];

        int maxLeft = 0;


        /**
         * 第i天以及i天之前 第一次交易最大值
         */
        for (int i = 1; i < prices.length; i++) {

            if (prices[i] < minPrice) {
                minPrice = prices[i];
            }
            maxLeft = Math.max(maxLeft, prices[i] - minPrice);

            left[i] = maxLeft;
        }


        int[] right = new int[length + 1];


        int maxRight = 0;


        minPrice = prices[length - 1];


        /**
         * 第i天以及i天之后 第一次交易最大值
         * 从右往左计算
         */
        for (int i = length - 2; i >= 0; i--) {

            if (prices[i] > minPrice) {

                minPrice = prices[i];
            }
            maxRight = Math.max(maxRight, minPrice - prices[i]);

            right[i] = maxRight;
        }


        int result = 0;


        for (int i = 0; i < length; i++) {
            result = Math.max(result, left[i] + right[i + 1]);
        }
        return result;
    }


    /**
     * 125. Valid Palindrome
     *
     * @param s
     * @return
     */
    public boolean isPalindrome(String s) {
        if (s == null) {
            return false;
        }

        s = s.trim();

        if (s.length() == 0) {
            return false;
        }

        int left = 0;

        int right = s.length() - 1;

        while (left < right) {
            while (left < right && !Character.isLetterOrDigit(s.charAt(left))) {
                left++;
            }
            while (left < right && !Character.isLetterOrDigit(s.charAt(right))) {
                right--;
            }
            if (Character.toLowerCase(s.charAt(left)) != Character.toLowerCase(s.charAt(right))) {
                return false;
            } else {
                left++;
                right--;
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
        int result = 0;

        /**
         * 方案一、遍历数组
         * 如果元素已经存在处理 故无需计算
         * 取出当前元素的 左边界 以及右边界
         * 获取其与左右边界的差值的最大值
         * 与全局值进行比较
         * 并且左右边界、当前元素的长度
         *
         */

//        HashMap<Integer, Integer> map = new HashMap<>();
//        for (int i = 0; i < nums.length; i++) {
//            if (map.containsKey(nums[i])) {
//                continue;
//            }
//            int left = map.getOrDefault(nums[i] - 1, 0);
//
//            int right = map.getOrDefault(nums[i] + 1, 0);
//
//            int sum = left + right + 1;
//
//            result = Math.max(result, left + right + 1);
//
//
//            map.put(nums[i], sum);
//            map.put(nums[i] - left, sum);
//            map.put(nums[i] + right, sum);
//
//        }
        Arrays.sort(nums);
        int current = 1;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] != nums[i - 1]) {
                if (nums[i] == nums[i - 1] + 1) {
                    current++;

                } else {
                    result = Math.max(result, current);
                    current = 1;
                }
            }

        }
        result = Math.max(result, current);
        /**
         * 方案二、允许排序的话 先排序
         * 排序之后在遍历处理
         */
        return result;
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


        /**
         * 深度优先遍历
         * 由于不需要查找操作
         * 故只需要满足二维数组边界条件即可
         * 根据当前目标进行深度遍历
         *
         */


        // 方案一
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                boolean edge = i == 0 || i == row - 1 || j == 0 || j == column - 1;
                if (edge && board[i][j] == 'O') {
                    this.solveDFS(board, i, j);
                }
            }
        }
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (board[i][j] == 'A') {
                    board[i][j] = 'O';
                } else if (board[i][j] == 'O') {
                    board[i][j] = 'X';
                }
            }
        }
    }

    private void solveDFS(char[][] board, int i, int j) {
        if (i < 0 || i >= board.length || j < 0 || j >= board[i].length || board[i][j] != 'O') {
            return;
        }
        board[i][j] = 'A';
        this.solveDFS(board, i - 1, j);
        this.solveDFS(board, i + 1, j);
        this.solveDFS(board, i, j - 1);
        this.solveDFS(board, i, j + 1);
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
        /**
         * 排序组合思路
         * 对于多纬list 可考虑使用 一维list存放数据
         * 满足题意 放进list里面
         */
        List<List<String>> ans = new ArrayList<>();
        this.partition(ans, new ArrayList<String>(), 0, s);

        return ans;
    }

    private void partition(List<List<String>> ans, List<String> tmp, int k, String s) {
        if (k == s.length()) {
            ans.add(new ArrayList<>(tmp));
        }
        for (int i = k; i < s.length(); i++) {
            if (this.isValid(s, k, i)) {
                tmp.add(s.substring(k, i + 1));
                this.partition(ans, tmp, i + 1, s);
                tmp.remove(tmp.size() - 1);
            }
        }
    }

    private boolean isValid(String s, int start, int end) {
        if (start > end) {
            return false;
        }
        while (start < end) {
            if (s.charAt(start) != s.charAt(end)) {
                return false;
            }
            start++;
            end--;
        }
        return true;
    }


    /**
     * 132. Palindrome Partitioning II
     * <p>
     * 动态规划问题
     * cut[j] = cut[j-1] + 1
     * if s.begin(j) == s.begin(i)
     * if j == 0 则 [0, i] 都是回文数字
     * 不用切割
     *
     * @param s
     * @return
     */
    public int minCut(String s) {
        if (s == null || s.length() == 0) {
            return 0;
        }

        int m = s.length();

        boolean[][] dp = new boolean[m][m];

        int[] cut = new int[m];
        for (int i = 0; i < m; i++) {
            int min = i;
            for (int j = 0; j <= i; j++) {
                if (s.charAt(j) == s.charAt(i) && (i - j < 2 || dp[j + 1][i - 1])) {
                    dp[j][i] = true;
                    min = j == 0 ? 0 : Math.min(min, cut[j - 1] + 1);
                }

            }
            cut[i] = min;
        }
        return cut[m - 1];
    }


    /**
     * 134. Gas Station
     *
     * @param gas
     * @param cost
     * @return
     */
    public int canCompleteCircuit(int[] gas, int[] cost) {
        if (gas == null || gas.length == 0) {
            return -1;
        }
        int global = 0;

        int local = 0;

        int current = 0;
        for (int i = 0; i < gas.length; i++) {

            global += gas[i] - cost[i];

            local += gas[i] - cost[i];

            if (local < 0) {
                current = i + 1;
            }
        }
        return current;
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
        int m = ratings.length;

        // 每个数组元素 需要与 其左右两边元素对比
        // 故需 遍历两次
        // 每个元素的糖果数量
        // 必须比左右两边多
        // 故需判断 dp[i] = 1 + dp[i-1]
        int[] dp = new int[m];
        for (int i = 0; i < m; i++) {
            dp[i] = 1;
        }
        for (int i = 1; i < m; i++) {
            if (ratings[i] > ratings[i - 1] && dp[i] < dp[i - 1] + 1) {
                dp[i] = 1 + dp[i - 1];
            }
        }
        for (int i = m - 2; i >= 0; i--) {
            if (ratings[i] > ratings[i + 1] && dp[i] < dp[i + 1] + 1) {
                dp[i] = 1 + dp[i + 1];
            }
        }
        int result = 0;
        for (int i = 0; i < m; i++) {
            result += dp[i];
        }
        return result;


    }

    /**
     * 136. Single Number
     */
    public int singleNumber(int[] nums) {
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
        Node current = head;
        while (current != null) {

            Node tmp = new Node(current.val, current.next, null);

            current.next = tmp;

            if (current.next != null) {

                current = tmp.next;
            }
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

        Node copyHead = head.next;

        while (current.next != null) {

            Node tmp = current.next;

            current.next = tmp.next;

            current = tmp;
        }

        return copyHead;


    }


    /**
     * 139. Word Break
     *
     * @param s
     * @param wordDict
     * @return
     */
    public boolean wordBreak(String s, List<String> wordDict) {
        if (s == null || s.isEmpty()) {
            return false;
        }
        // 深度优先遍历思想

        // 由于可能存在超时问题

        // 需要减少计算次数

        // 故需存储已经计算的数据

        HashMap<String, Boolean> included = new HashMap<>();
        return this.dfsWord(included, s, wordDict);

    }

    private boolean dfsWord(HashMap<String, Boolean> included, String s, List<String> wordDict) {
        if (wordDict.contains(s)) {
            return true;
        }
        if (included.containsKey(s)) {
            return false;
        }
        for (String word : wordDict) {
            if (s.startsWith(word) && this.dfsWord(included, s.substring(word.length()), wordDict)) {
                return true;
            }
        }
        included.put(s, false);
        return false;
    }


    /**
     * 140. Word Break II
     * <p>
     * todo 思路 深度优先遍历
     *
     * @param s
     * @param wordDict
     * @return
     */
    public List<String> wordBreakII(String s, List<String> wordDict) {
        if (s == null || s.length() == 0) {
            return new ArrayList<>();
        }
        HashMap<String, List<String>> map = new HashMap<>();
        return this.wordBreakII(map, s, wordDict);
    }

    private List<String> wordBreakII(HashMap<String, List<String>> map, String s, List<String> wordDict) {
        if (map.containsKey(s)) {
            return map.get(s);
        }
        List<String> ans = new ArrayList<>();
        if (s.length() == 0) {
            ans.add("");
            return ans;
        }
        for (String word : wordDict) {
            if (s.startsWith(word)) {
                List<String> tmp = this.wordBreakII(map, s.substring(word.length()), wordDict);

                for (String value : tmp) {
                    String str = value.isEmpty() ? "" : " ";
                    ans.add(word + str);
                }
            }
        }

        map.put(s, ans);
        return ans;
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
        ListNode fast = head;
        ListNode slow = head;
        while (fast.next != null && fast.next.next != null) {

            fast = fast.next.next;

            slow = slow.next;
            if (fast == slow) {
                return true;
            }
        }
        return false;


    }


    /**
     * 142. Linked List Cycle II
     *
     * @param head
     * @return
     */
    public ListNode detectCycle(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode slow = head;
        ListNode fast = head;
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (fast == slow) {
                fast = head;
                while (fast != slow) {
                    fast = fast.next;
                    slow = slow.next;
                }
                return fast;

            }
        }
        return null;
    }


    /**
     * 143. Reorder List
     *
     * @param head
     */
    public void reorderList(ListNode head) {
        if (head == null || head.next == null) {
            return;
        }
        ListNode fast = head;

        ListNode slow = head;

        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }

        ListNode middle = slow;

        /**
         *
         */
        ListNode current = slow.next;


        ListNode prev = this.reverse(current, null);


        slow.next = prev;


        slow = head;


        fast = middle.next;


        while (slow != middle) {
            middle.next = fast.next;

            fast.next = slow.next;

            slow.next = fast;


            slow = fast.next;

            fast = middle.next;

        }

    }

    private ListNode reverse(ListNode start, ListNode last) {
        ListNode end = last;


        while (start != last) {


            ListNode tmp = start.next;


            start.next = end;

            end = start;

            start = tmp;
        }
        return end;
    }


    /**
     * 144. Binary Tree Preorder Traversal
     *
     * @param root
     * @return
     */
    public List<Integer> preorderTraversal(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        List<Integer> ans = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        while (!stack.isEmpty()) {
            root = stack.pop();
            if (root.right != null) {
                stack.push(root.right);
            }
            if (root.left != null) {
                stack.push(root.left);
            }
            ans.add(root.val);
        }
        return ans;
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
        LinkedList<Integer> ans = new LinkedList<>();
        TreeNode node = root;
        Stack<TreeNode> stack = new Stack<>();
        while (!stack.isEmpty() || node != null) {
            if (node != null) {
                stack.push(node);
                ans.addFirst(node.val);
                node = node.right;
            } else {
                node = stack.pop();
                node = node.left;

            }
        }
        return ans;
    }
}
