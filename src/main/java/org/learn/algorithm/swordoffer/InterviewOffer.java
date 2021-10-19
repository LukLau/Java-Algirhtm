package org.learn.algorithm.swordoffer;

import org.learn.algorithm.datastructure.Interval;
import org.learn.algorithm.datastructure.ListNode;
import org.learn.algorithm.datastructure.TreeNode;

import java.util.*;

/**
 * 剑指offer
 *
 * @author luk
 * @date 2021/5/2
 */
public class InterviewOffer {

    /**
     * @param root TreeNode类
     * @return int整型
     */
    private int maxResult = Integer.MIN_VALUE;

    public static void main(String[] args) {
        InterviewOffer interviewOffer = new InterviewOffer();
        String param = "(3+4)*(5+(2-3))";
        interviewOffer.calculatorII(param);
    }

    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * 计算两个数之和
     *
     * @param s string字符串 表示第一个整数
     * @param t string字符串 表示第二个整数
     * @return string字符串
     */
    public String solve(String s, String t) {
        // write code here
        if (s == null || t == null) {
            return "";
        }
        char[] s1 = s.toCharArray();
        char[] t1 = t.toCharArray();
        int carry = 0;
        int m = s1.length - 1;
        int n = t1.length - 1;
        StringBuilder builder = new StringBuilder();
        while (m >= 0 || n >= 0 || carry > 0) {
            int val = (m >= 0 ? Character.getNumericValue(s1[m--]) : 0) + (n >= 0 ? Character.getNumericValue(t1[n--]) : 0) + carry;
            builder.append(val % 10);
            carry = val / 10;
        }
        return builder.reverse().toString();
    }

    /**
     * NC92 最长公共子序列-II
     * longest common subsequence
     *
     * @param s1 string字符串 the string
     * @param s2 string字符串 the string
     * @return string字符串
     */
    public String longestCommonSubsequence(String s1, String s2) {
        // write code here
        if (s1 == null || s2 == null) {
            return "-1";
        }
        int m = s1.length();
        int n = s2.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (s1.charAt(i - 1) == s2.charAt(j - 1)) {
                    dp[i][j] = 1 + dp[i - 1][j - 1];
                } else {
                    dp[i][j] = Math.max(dp[i][j - 1], dp[i - 1][j]);
                }
            }
        }
        StringBuilder builder = new StringBuilder();
        while (m >= 1 && n >= 1) {
            if (s1.charAt(m - 1) == s2.charAt(n - 1)) {
                builder.append(s1.charAt(m - 1));
                m--;
                n--;
            } else if (dp[m][n - 1] > dp[m - 1][n]) {
                n--;
            } else {
                m--;
            }
        }
        if (builder.length() == 0) {
            return "-1";
        }
        return builder.reverse().toString();
    }


    /**
     * 最长公共字串
     * longest common substring
     *
     * @param str1 string字符串 the string
     * @param str2 string字符串 the string
     * @return string字符串
     */
    public String LCS(String str1, String str2) {
        // write code here
        if (str1 == null || str2 == null) {
            return "";
        }
        int m = str1.length();
        int n = str2.length();
        int result = 0;
        int index = 0;
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (str1.charAt(i - 1) == str2.charAt(j - 1)) {
                    dp[i][j] = 1 + dp[i - 1][j - 1];
                    if (dp[i][j] > result) {
                        result = dp[i][j];
                        index = i;
                    }
                }
            }
        }
        if (result == 0) {
            return "";
        }
        return str1.substring(index - result, index);
    }

    /**
     * 反转字符串
     *
     * @param str string字符串
     * @return string字符串
     */
    public String solve(String str) {
        // write code here
        if (str == null || str.isEmpty()) {
            return "";
        }
        char[] words = str.toCharArray();
        reverse(words, 0, words.length - 1);
        return String.valueOf(words);
    }

    private void reverse(char[] words, int start, int end) {
        for (int i = start; i <= (start + end) / 2; i++) {
            swap(words, i, start + end - i);
        }
    }

    private void swap(char[] words, int i, int j) {
        char tmp = words[i];
        words[i] = words[j];
        words[j] = tmp;
    }

    /**
     * @param x int整型
     * @return int整型
     */
    public int reverse(int x) {
        if (x == 0) {
            return 0;
        }
        int sign = x < 0 ? -1 : 1;
        int result = 0;
        while (x != 0) {
            if (x > Integer.MAX_VALUE / 10) {
                return 0;
            }
            result = result * 10 + x % 10;
            x /= 10;
        }
        return result * sign;
        // write code here
    }

    /**
     * NC35 最小编辑代价
     * todo
     * min edit cost
     *
     * @param str1 string字符串 the string
     * @param str2 string字符串 the string
     * @param ic   int整型 insert cost
     * @param dc   int整型 delete cost
     * @param rc   int整型 replace cost
     * @return int整型
     */
    public int minEditCost(String str1, String str2, int ic, int dc, int rc) {
        // write code here
        if (str1 == null || str2 == null) {
            return 0;
        }
        int m = str1.length();
        int n = str2.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++) {
            dp[i][0] = dp[i - 1][0] + ic;
        }
        for (int j = 1; j <= n; j++) {
            dp[0][j] = dp[0][j - 1] + dc;
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (str1.charAt(i - 1) == str2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = Math.max(Math.max(dp[i - 1][j] + dc, dp[i][j - 1] + dc), dp[i - 1][j - 1] + rc);
                }
            }
        }
        return dp[m][n];
    }

    public int solve(char[][] grid) {
        // write code here
        if (grid == null || grid.length == 0) {
            return 0;
        }
        int row = grid.length;
        int column = grid[0].length;
        int count = 0;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (grid[i][j] == '1') {
                    count++;
                    intervalDFSSolve(grid, i, j);
                }
            }
        }
        return count;
    }

    private void intervalDFSSolve(char[][] grid, int i, int j) {
        if (i < 0 || i == grid.length || j < 0 || j == grid[i].length || grid[i][j] != '1') {
            return;
        }
        grid[i][j] = '0';
        intervalDFSSolve(grid, i - 1, j);
        intervalDFSSolve(grid, i + 1, j);
        intervalDFSSolve(grid, i, j - 1);
        intervalDFSSolve(grid, i, j + 1);
    }

    /**
     * NC66 两个链表的第一个公共结点
     *
     * @param pHead1
     * @param pHead2
     * @return
     */
    public ListNode FindFirstCommonNode(ListNode pHead1, ListNode pHead2) {
        ListNode p1 = pHead1;
        ListNode p2 = pHead2;
        while (p1 != p2) {
            p1 = p1 == null ? pHead2 : p1.next;
            p2 = p2 == null ? pHead1 : p2.next;
        }
        return p1;
    }

    /**
     * todo
     * retrun the longest increasing subsequence
     *
     * @param arr int整型一维数组 the array
     * @return int整型一维数组
     */
    public int[] LIS(int[] arr) {
        // write code here
        if (arr == null || arr.length == 0) {
            return new int[]{};
        }
        int n = arr.length;
        int[] dp = new int[n];
        Arrays.fill(dp, 1);
        int result = 0;
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (arr[j] < arr[i] && dp[i] < dp[j] + 1) {
                    dp[i] = dp[j] + 1;
                    result = Math.max(result, dp[i]);
                }
            }
        }
        int[] ans = new int[result];
        for (int i = arr.length - 1; i >= 0; i--) {
        }
        return null;
    }

    /**
     * NC37 合并区间
     *
     * @param intervals
     * @return
     */
    public ArrayList<Interval> merge(ArrayList<Interval> intervals) {
        if (intervals == null || intervals.isEmpty()) {
            return new ArrayList<>();
        }
        intervals.sort(Comparator.comparingInt(o -> o.start));
        ArrayList<Interval> result = new ArrayList<>();
        result.add(intervals.get(0));
        int size = intervals.size();
        int index = 0;
        for (int i = 1; i < size; i++) {
            Interval interval = intervals.get(i);
            if (result.get(index).end < interval.start) {
                result.add(interval);
                index++;
            } else {
                Interval pre = result.get(index);
                pre.start = Math.min(pre.start, interval.start);
                pre.end = Math.max(pre.end, interval.end);
            }
        }
        return result;
    }

    /**
     * @param str string字符串
     * @return int整型
     */
    public int atoi(String str) {
        // write code here
        if (str == null) {
            return 0;
        }
        str = str.trim();
        if (str.isEmpty()) {
            return 0;
        }

        int index = 0;
        int sign = 1;
        char[] words = str.toCharArray();
        if (words[index] == '+' || words[index] == '-') {
            sign = words[index] == '+' ? 1 : -1;
            index++;
        }
        long result = 0;
        while (index < words.length && Character.isDigit(words[index])) {
            result = result * 10 + Character.getNumericValue(words[index]);
            if (result > Integer.MAX_VALUE) {
                return sign == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
            }
            index++;
        }
        return (int) (result * sign);
    }

    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * <p>
     * 找缺失数字
     *
     * @param a int整型一维数组 给定的数字串
     * @return int整型
     */
    public int solve(int[] a) {
        // write code here
        return -1;
    }

    /**
     * todo
     *
     * @param root TreeNode类 the root
     * @return bool布尔型一维数组
     */
    public boolean[] judgeIt(TreeNode root) {
        // write code here
        if (root == null) {
            return new boolean[]{};
        }
        boolean[] result = new boolean[2];
        result[0] = checkSearch(root, Integer.MIN_VALUE, Integer.MAX_VALUE);
        result[1] = checkBalance(root);
        return result;
    }

    private boolean checkSearch(TreeNode root, int min, int max) {
        if (root == null) {
            return true;
        }
        if (root.val <= min || root.val >= max) {
            return false;
        }
        return checkSearch(root.left, min, root.val) && checkSearch(root.right, root.val, max);
    }

    private boolean checkBalance(TreeNode root) {
        if (root == null) {
            return true;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode node = queue.poll();
                //如果当前节点并不是左右孩子节点全有，那么之后的节点必须都为叶节点
                if (node.left == null && node.right != null) {
                    return false;
                }
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
            }
        }
        return true;
    }

    private int maxDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
    }

    /**
     * NC134 股票(无限次交易)
     *
     * @param prices int整型一维数组
     * @return int整型
     */
    public int maxProfit(int[] prices) {
        if (prices == null || prices.length == 0) {
            return 0;
        }
        int result = 0;
        int cost = prices[0];
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > cost) {
                result += prices[i] - cost;
            }
            cost = prices[i];
        }
        return result;

    }

    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param nums   int整型一维数组
     * @param target int整型
     * @return int整型
     */
    public int search(int[] nums, int target) {
        // write code here
        if (nums == null || nums.length == 0) {
            return -1;
        }
        int left = 0;
        int right = nums.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[left] <= nums[mid]) {
                if (target >= nums[left] && target < nums[mid]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            } else {
                if (target > nums[mid] && target <= nums[right]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }
        return nums[left] == target ? left : -1;
    }

    /**
     * todo
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * 求二叉树的右视图
     *
     * @param xianxu  int整型一维数组 先序遍历
     * @param zhongxu int整型一维数组 中序遍历
     * @return int整型一维数组
     */
    public int[] solve(int[] xianxu, int[] zhongxu) {
        // write code here
        if (xianxu == null || zhongxu == null) {
            return new int[]{};
        }
        return new int[]{};
    }

    /**
     * 验证IP地址
     *
     * @param IP string字符串 一个IP地址字符串
     * @return string字符串
     */
    public String solveIP(String IP) {
        // write code here
        if (IP == null || IP.isEmpty()) {
            return "Neither";
        }
        if (IP.indexOf(".") > 0) {
            String[] split = IP.split("\\.");
            if (split.length != 4) {
                return "Neither";
            }
            for (String word : split) {
                if (!checkIpV4(word)) {
                    return "Neither";
                }
            }
            return "IPv4";
        }
        String[] words = IP.split(":");
        if (words.length != 8) {
            return "Neither";
        }
        for (String word : words) {
            int length = word.length();
            if (length <= 0) {
                return "Neither";
            }
            if ("000".equals(word) || "0000".equals(word) || "00".equals(word)) {
                return "Neither";
            }
        }
        return "IPv6";
    }

    private boolean checkIpV4(String word) {
        int length = word.length();
        if (length <= 0 || length > 3) {
            return false;
        }
        int num = Integer.parseInt(word);
        if (num > 255 || num < 0) {
            return false;
        }
        return length <= 2 || word.charAt(0) != '0';
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
     * NC38 螺旋矩阵
     *
     * @param matrix
     * @return
     */
    public ArrayList<Integer> spiralOrder(int[][] matrix) {
        ArrayList<Integer> result = new ArrayList<>();
        if (matrix == null || matrix.length == 0) {
            return result;
        }
        int row = matrix.length;
        int column = matrix[0].length;
        int top = 0;
        int bottom = row - 1;
        int left = 0;
        int right = column - 1;
        while (left <= right && top <= bottom) {
            for (int i = left; i <= right; i++) {
                result.add(matrix[top][i]);
            }
            for (int i = top + 1; i <= bottom; i++) {
                result.add(matrix[i][right]);
            }
            if (top != bottom) {
                for (int i = right - 1; i >= left; i--) {
                    result.add(matrix[bottom][i]);
                }
            }
            if (left != right) {
                for (int i = bottom - 1; i > top; i--) {
                    result.add(matrix[i][left]);
                }
            }
            left++;
            right--;
            top++;
            bottom--;
        }
        return result;
    }


    /**
     * WC127 滑动窗口的最大值
     *
     * @param num
     * @param size
     * @return
     */
    public ArrayList<Integer> maxInWindows(int[] num, int size) {
        if (num == null || num.length == 0 || size == 0) {
            return new ArrayList<>();
        }
        LinkedList<Integer> linkedList = new LinkedList<>();
        ArrayList<Integer> result = new ArrayList<>();
        for (int i = 0; i < num.length; i++) {
            int index = i - size + 1;
            if (!linkedList.isEmpty() && linkedList.peek() < index) {
                linkedList.poll();
            }
            while (!linkedList.isEmpty() && num[linkedList.peekLast()] <= num[i]) {
                linkedList.pollLast();
            }
            linkedList.offer(i);
            if (index >= 0) {
                result.add(num[linkedList.peekFirst()]);
            }
        }
        return result;
    }

    /**
     * @param strs string字符串一维数组
     * @return string字符串
     */
    public String longestCommonPrefix(String[] strs) {
        // write code here
        if (strs == null || strs.length == 0) {
            return "";
        }
        String prefix = strs[0];
        for (int i = 1; i < strs.length; i++) {
            while (strs[i].indexOf(prefix) != 0) {
                prefix = prefix.substring(0, prefix.length() - 1);
            }
        }
        return prefix;
    }

    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * 将给定数组排序
     *
     * @param arr int整型一维数组 待排序的数组
     * @return int整型一维数组
     */
    public int[] MySort(int[] arr) {
        // write code here
        if (arr == null || arr.length == 0) {
            return arr;
        }
        quickSort(arr, 0, arr.length - 1);
        return arr;
    }

    private void quickSort(int[] nums, int start, int end) {
        if (start > end) {
            return;
        }
        int partition = getPartition(nums, start, end);
        quickSort(nums, start, partition - 1);
        quickSort(nums, partition + 1, end);
    }

    private int getPartition(int[] nums, int start, int end) {
        int pivot = nums[start];
        while (start < end) {
            while (start < end && nums[end] >= pivot) {
                end--;
            }
            if (start < end) {
                swap(nums, start, end);
                start++;
            }
            while (start < end && nums[start] <= pivot) {
                start++;
            }
            if (start < end) {
                swap(nums, start, end);
                end--;
            }
        }
        nums[start] = pivot;
        return start;
    }

    private void swap(int[] nums, int start, int end) {
        int val = nums[start];
        nums[start] = nums[end];
        nums[end] = val;
    }

    public TreeNode reConstructBinaryTree(int[] pre, int[] in) {
        if (pre == null || in == null) {
            return null;
        }
        return buildTree(0, pre, 0, in.length - 1, in);
    }

    private TreeNode buildTree(int preStart, int[] pre, int inStart, int inEnd, int[] in) {
        if (preStart == pre.length || inStart > inEnd) {
            return null;
        }
        int val = pre[preStart];
        int index = 0;
        for (int i = inStart; i <= inEnd; i++) {
            if (in[i] == val) {
                index = i;
                break;
            }
        }
        TreeNode root = new TreeNode(val);
        root.left = buildTree(preStart + 1, pre, inStart, index - 1, in);
        root.right = buildTree(preStart + index - inStart + 1, pre, index + 1, inEnd, in);
        return root;
    }

    /**
     * @param root TreeNode类 the root of binary tree
     * @return int整型二维数组
     */
    public int[][] threeOrders(TreeNode root) {
        // write code here
        if (root == null) {
            return new int[][]{};
        }
        return null;
    }

    public String trans(String s, int n) {
        // write code here
        if (s == null || n <= 0) {
            return "";
        }
        String[] words = s.split(" ", -1);
        StringBuilder builder = new StringBuilder();
        for (int i = words.length - 1; i >= 0; i--) {
            String word = words[i];
            builder.append(checkTrans(word.toCharArray()));
            if (i != 0) {
                builder.append(" ");
            }
        }
        return builder.toString();
    }

    private String checkTrans(char[] str) {
        for (int i = 0; i < str.length; i++) {
            char tmp = str[i];
            if (Character.isLetter(tmp)) {
                if (Character.isLowerCase(tmp)) {
                    str[i] = Character.toUpperCase(tmp);
                } else {
                    str[i] = Character.toLowerCase(tmp);
                }
            }
        }
        return String.valueOf(str);
    }

    /**
     * todo
     * 树的直径
     *
     * @param n          int整型 树的节点个数
     * @param Tree_edge  Interval类一维数组 树的边
     * @param Edge_value int整型一维数组 边的权值
     * @return int整型
     */
    public int solve(int n, Interval[] Tree_edge, int[] Edge_value) {
        // write code here
        return -1;
    }

    /**
     * @param n int整型
     * @param m int整型
     * @return int整型
     */
    public int ysf(int n, int m) {
        // write code here
        List<Integer> tmp = new ArrayList<>();
        for (int i = 1; i <= n; i++) {
            tmp.add(i);
        }
        int current = 0;
        while (tmp.size() > 1) {
            int size = tmp.size();
            int index = (current + m - 1) % size;
            tmp.remove(index);
            current = index % (size - 1);
        }
        return tmp.get(0);
    }

    public boolean IsBalanced_Solution(TreeNode root) {
        if (root == null) {
            return true;
        }
        int left = maxDepth(root.left);
        int right = maxDepth(root.right);
        if (Math.abs(left - right) > 1) {
            return false;
        }
        return IsBalanced_Solution(root.left) && IsBalanced_Solution(root.right);
    }

    /**
     * https://www.nowcoder.com/practice/1af528f68adc4c20bf5d1456eddb080a?tpId=188&tags=&title=&diffculty=0&judgeStatus=0&rp=1&tab=answerKey
     * todo
     * 寻找最后的山峰
     *
     * @param a int整型一维数组
     * @return int整型
     */
    public int getPeek(int[] a) {
        // write code here
        if (a == null || a.length == 0) {
            return -1;
        }
        int left = 0;
        int right = a.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (a[mid] < a[mid + 1]) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return left;
    }

    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * 计算01背包问题的结果
     *
     * @param V  int整型 背包的体积
     * @param n  int整型 物品的个数
     * @param vw int整型二维数组 第一维度为n,第二维度为2的二维数组,vw[i][0],vw[i][1]分别描述i+1个物品的vi,wi
     * @return int整型
     */
    public int knapsack(int V, int n, int[][] vw) {
        // write code here
        if (vw == null || vw.length == 0) {
            return 0;
        }
        int[] dp = new int[V + 1];
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= V; j++) {
                if (j - vw[i - 1][0] >= 0) {
                    dp[j] = Math.max(dp[j], dp[j - vw[i - 1][0]] + vw[i - 1][1]);
                } else {
                    dp[j] = dp[j];
                }
            }
        }
        return dp[V];
    }


    /**
     * NC151 最大公约数
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * <p>
     * 求出a、b的最大公约数。
     *
     * @param a int
     * @param b int
     * @return int
     */
    public int gcd(int a, int b) {
        // write code here
        if (b == 0) {
            return a;
        }
        return gcd(b, a % b);
    }

    /**
     * NC128 接雨水问题
     *
     * @param arr
     * @return
     */
    public long maxWater(int[] arr) {
        // write code here
        if (arr == null || arr.length == 0) {
            return 0;
        }
        int left = 0;
        int right = arr.length - 1;
        long result = 0;
        int leftEdge = 0;
        int rightEdge = 0;
        while (left < right) {
            if (arr[left] <= arr[right]) {
                if (arr[left] >= leftEdge) {
                    leftEdge = arr[left];
                } else {
                    result += leftEdge - arr[left];
                }
                left++;
            } else {
                if (arr[right] >= rightEdge) {
                    rightEdge = arr[right];
                } else {
                    result += rightEdge - arr[right];
                }
                right--;
            }
        }
        return result;
    }

    public boolean isMatch(String s, String p) {
        if (p.isEmpty()) {
            return s.isEmpty();
        }
        int m = s.length();
        int n = p.length();
        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[0][0] = true;
        for (int i = 1; i <= n; i++) {
            dp[0][i] = p.charAt(i - 1) == '*' && dp[0][i - 1];
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (s.charAt(i - 1) == p.charAt(j - 1) || p.charAt(j - 1) == '?') {
                    dp[i][j] = dp[i - 1][j - 1];
                } else if (p.charAt(j - 1) == '*') {
                    dp[i][j] = dp[i - 1][j] || dp[i][j - 1];
                }
            }
        }
        return dp[m][n];
    }

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
        ListNode next = slow.next;

        ListNode reverseNode = reverseNode(next);

        slow.next = null;

        ListNode d1 = head;
        while (d1 != null && reverseNode != null) {
            ListNode tmp = d1.next;

            d1.next = reverseNode;

            ListNode reverse2 = reverseNode.next;

            reverseNode.next = tmp;

            d1 = reverseNode.next;

            reverseNode = reverse2;
        }
    }

    private ListNode reverseNode(ListNode head) {
        ListNode prev = null;
        while (head != null) {
            ListNode tmp = head.next;
            head.next = prev;
            prev = head;
            head = tmp;
        }
        return prev;
    }

    /**
     * @param root TreeNode类
     * @return int整型
     */
    public int sumNumbers(TreeNode root) {
        // write code here
        if (root == null) {
            return 0;
        }
        return intervalSum(root, 0);

    }

    private int intervalSum(TreeNode root, int sum) {
        if (root == null) {
            return 0;
        }
        if (root.left == null && root.right == null) {
            return sum * 10 + root.val;
        }
        return intervalSum(root.left, sum * 10 + root.val) + intervalSum(root.right, sum * 10 + root.val);
    }

    /**
     * todo
     * return topK string
     *
     * @param strings string字符串一维数组 strings
     * @param k       int整型 the k
     * @return string字符串二维数组
     */
    public String[][] topKstrings(String[] strings, int k) {
        // write code here
        if (strings == null || strings.length == 0) {
            return new String[][]{};
        }
        Map<String, Integer> map = new HashMap<>();
        for (String word : strings) {
            Integer count = map.getOrDefault(word, 0);
            count++;
            map.put(word, count);
        }
        PriorityQueue<String> queue = new PriorityQueue<>(
                (w1, w2) -> map.get(w1).equals(map.get(w2)) ?
                        w2.compareTo(w1) : map.get(w1) - map.get(w2));

        String[][] result = new String[k][2];
        for (Map.Entry<String, Integer> item : map.entrySet()) {
            String key = item.getKey();
            queue.offer(key);
            if (queue.size() > k) {
                queue.poll();
            }
        }
        int iterator = k - 1;
        while (!queue.isEmpty()) {
            String tmp = queue.poll();
            result[iterator][0] = tmp;
            result[iterator][1] = map.get(tmp) + "";
            iterator--;
        }
        return result;
    }

    /**
     * todo
     * NC156 数组中只出现一次的数（其它数出现k次）
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param arr int一维数组
     * @param k   int
     * @return int
     */
    public int foundOnceNumber(int[] arr, int k) {
        // write code here
        return -1;
    }

    /**
     * todo
     *
     * @param array
     * @return
     */
    public int InversePairs(int[] array) {
        return -1;
    }

    public int maxPathSum(TreeNode root) {
        // write code here

        if (root == null) {
            return 0;
        }
        dfsMaxPathSum(root);
        return maxResult;

    }

    private int dfsMaxPathSum(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int left = dfsMaxPathSum(root.left);
        int right = dfsMaxPathSum(root.right);
        maxResult = Math.max(maxResult, left + right + root.val);
        return Math.max(left, right) + root.val;
    }


    /**
     * NC83 子数组最大乘积
     *
     * @param arr
     * @return
     */
    public double maxProduct(double[] arr) {
        double max = arr[0];

        double min = arr[0];

        double result = arr[0];

        for (int i = 1; i < arr.length; i++) {
            double val = arr[i];
            double tmpMax = Math.max(Math.max(max * val, min * val), val);
            double tmpMin = Math.min(Math.min(max * val, min * val), val);

            result = Math.max(result, tmpMax);

            max = tmpMax;

            min = tmpMin;
        }
        return result;
    }


    /**
     * @param numbers int整型一维数组
     * @param target  int整型
     * @return int整型一维数组
     */
    public int[] twoSum(int[] numbers, int target) {
        // write code here
        if (numbers == null || numbers.length == 0) {
            return new int[]{-1, -1};
        }
        int[] result = new int[]{-1, -1};
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < numbers.length; i++) {
            if (map.containsKey(target - numbers[i])) {
                result[0] = map.get(target - numbers[i]) + 1;
                result[1] = i + 1;
                return result;
            }
            map.put(numbers[i], i);
        }
        return result;
    }


    /**
     * NC41 最长无重复子数组
     *
     * @param arr int整型一维数组 the array
     * @return int整型
     */
    public int maxLength(int[] arr) {
        // write code here
        if (arr == null || arr.length == 0) {
            return 0;
        }
        int left = 0;
        int result = 0;
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < arr.length; i++) {
            if (map.containsKey(arr[i])) {
                left = Math.max(left, map.get(arr[i]) + 1);
            }
            map.put(arr[i], i);
            result = Math.max(result, i - left + 1);
        }
        return result;
    }
    /**
     * todo 无重复子递增数组
     */


    /**
     * @param s string字符串
     * @return bool布尔型
     */
    public boolean isValid(String s) {
        // write code here
        if (s == null || s.isEmpty()) {
            return false;
        }
        Stack<Character> stack = new Stack<>();

        char[] words = s.toCharArray();

        for (char word : words) {
            if (word == '(') {
                stack.push(')');
            } else if (word == '[') {
                stack.push(']');
            } else if (word == '{') {
                stack.push('}');
            } else {
                if (stack.isEmpty() || stack.peek() != word) {
                    return false;
                }
                stack.pop();
            }
        }
        return stack.isEmpty();
    }


    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * <p>
     * 如果目标值存在返回下标，否则返回 -1
     *
     * @param nums   int整型一维数组
     * @param target int整型
     * @return int整型
     */
    public int searchII(int[] nums, int target) {
        // write code here
        if (nums == null || nums.length == 0) {
            return -1;
        }
        int left = 0;

        int right = nums.length - 1;

        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return nums[left] == target ? left : -1;
    }


    /**
     * NC17 最长回文子串
     *
     * @param A
     * @param n
     * @return
     */
    public int getLongestPalindrome(String A, int n) {
        // write code here
        if (A == null || A.length() == 0) {
            return 0;
        }
        boolean[][] dp = new boolean[n][n];
        int result = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                if (A.charAt(j) == A.charAt(i) && (i - j <= 2 || dp[j + 1][i - 1])) {
                    dp[j][i] = true;
                }
                if (dp[j][i] && i - j + 1 > result) {
                    result = i - j + 1;
                }
            }
        }
        return result;
    }

    public int MoreThanHalfNum_Solution(int[] array) {
        if (array == null || array.length == 0) {
            return 0;
        }
        int candidate = array[0];
        int count = 0;
        for (int num : array) {
            if (num == candidate) {
                count++;
            } else {
                count--;
                if (count == 0) {
                    candidate = num;
                    count++;
                }
            }
        }
        count = 0;
        for (int num : array) {
            if (num == candidate) {
                count++;
            }
        }
        return 2 * count >= array.length ? candidate : -1;
    }


    public ListNode detectCycle(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode slow = head;
        ListNode fast = head;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (slow == fast) {
                fast = head;
                while (fast != slow) {
                    fast = fast.next;
                    slow = slow.next;
                }
                return slow;
            }
        }
        return null;
    }


    /**
     * NC59 矩阵的最小路径和
     *
     * @param matrix
     * @return
     */
    public int minPathSum(int[][] matrix) {
        // write code here
        if (matrix == null || matrix.length == 0) {
            return 0;
        }
        int row = matrix.length;
        int column = matrix[0].length;
        int[] dp = new int[column];
        for (int j = 0; j < column; j++) {
            dp[j] = matrix[0][j] + (j == 0 ? 0 : dp[j - 1]);
        }
        for (int i = 1; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (j == 0) {
                    dp[j] += matrix[i][j];
                } else {
                    dp[j] = Math.min(dp[j], dp[j - 1]) + matrix[i][j];
                }
            }
        }
        return dp[column - 1];
    }

    public boolean hasCycle(ListNode head) {
        if (head == null || head.next == null) {
            return false;
        }
        ListNode fast = head;
        ListNode slow = head;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (fast == slow) {
                return true;
            }
        }
        return false;
    }


    public void merge(int A[], int m, int B[], int n) {
        int k = m + n;
        m--;
        n--;
        k--;
        while (m >= 0 && n >= 0) {
            if (A[m] > B[n]) {
                A[k--] = A[m--];
            } else {
                A[k--] = B[n--];
            }
        }
        while (n >= 0) {
            A[k--] = B[n--];
        }
    }


    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param pHead ListNode类
     * @param k     int整型
     * @return ListNode类
     */
    public ListNode FindKthToTail(ListNode pHead, int k) {
        // write code here
        if (pHead == null || k <= 0) {
            return null;
        }
        ListNode fast = pHead;
        int count = 1;
        while (fast.next != null) {
            count++;
            fast = fast.next;
        }
        return fast;
    }


    /**
     * @param l1 ListNode类
     * @param l2 ListNode类
     * @return ListNode类
     */
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        // write code here
        if (l1 == null && l2 == null) {
            return null;
        }
        if (l1 == null) {
            return l2;
        }
        if (l2 == null) {
            return l1;
        }
        if (l1.val <= l2.val) {
            l1.next = mergeTwoLists(l1.next, l2);
            return l1;
        } else {
            l2.next = mergeTwoLists(l1, l2.next);
            return l2;
        }
    }


    public ArrayList<ArrayList<Integer>> threeSum(int[] num) {
        if (num == null || num.length < 3) {
            return new ArrayList<>();
        }
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        Arrays.sort(num);
        for (int i = 0; i < num.length - 2; i++) {
            if (i > 0 && num[i] == num[i - 1]) {
                continue;
            }
            int left = i + 1;
            int right = num.length - 1;
            while (left < right) {
                int val = num[i] + num[left] + num[right];
                if (val == 0) {
                    result.add(new ArrayList<>(Arrays.asList(num[i], num[left], num[right])));
                    while (left < right && num[left] == num[left + 1]) {
                        left++;
                    }
                    while (left < right && num[right] == num[right - 1]) {
                        right--;
                    }
                    left++;
                    right--;
                } else if (val < 0) {
                    left++;
                } else {
                    right--;
                }
            }
        }
        return result;
    }


    public int findKth(int[] a, int n, int K) {
        // write code here'
        K--;
        int partition = getPartition(a, 0, n - 1);
        while (partition != n - K) {
            if (partition > K) {
                partition = getPartition(a, 0, partition - 1);
            } else {
                partition = getPartition(a, partition + 1, n - 1);
            }
        }
        return a[K];
    }


    /**
     * max sum of the subarray
     *
     * @param arr int整型一维数组 the array
     * @return int整型
     */
    public int maxsumofSubarray(int[] arr) {
        // write code here
        if (arr == null || arr.length == 0) {
            return 0;
        }
        int local = 0;
        int global = 0;
        for (int num : arr) {
            local = local >= 0 ? local + num : num;
            global = Math.max(local, global);
        }
        return global;
    }


    /**
     * @param head ListNode类
     * @param n    int整型
     * @return ListNode类
     */
    public ListNode removeNthFromEnd(ListNode head, int n) {
        // write code here
        ListNode root = new ListNode(0);
        root.next = head;
        ListNode fast = root;
        for (int i = 0; i < n; i++) {
            fast = fast.next;
        }
        ListNode slow = root;
        while (fast.next != null) {
            slow = slow.next;
            fast = fast.next;
        }
        ListNode next = slow.next;
        slow.next = next.next;
        next.next = null;
        return root.next;
    }


    /**
     * NC86 矩阵元素查找
     *
     * @param mat
     * @param n
     * @param m
     * @param x
     * @return
     */
    public int[] findElement(int[][] mat, int n, int m, int x) {
        // write code here
        int i = n - 1;
        int j = 0;
        while (i >= 0 && j < m) {
            int val = mat[i][j];
            if (val == x) {
                return new int[]{i, j};
            } else if (val < x) {
                j++;
            } else {
                i--;
            }
        }
        return new int[]{-1, -1};
    }


    public int NumberOf1(int n) {
        int count = 0;
        while (n != 0) {
            count++;
            n &= n - 1;
        }
        return count;
    }

    /**
     * @param n int整型
     * @return string字符串ArrayList
     */
    public ArrayList<String> generateParenthesis(int n) {
        // write code here
        if (n <= 0) {
            return new ArrayList<>();
        }
        ArrayList<String> result = new ArrayList<>();
        generate(result, 0, 0, "", n);
        return result;
    }

    private void generate(ArrayList<String> result, int open, int close, String s, int n) {
        if (s.length() == 2 * n) {
            result.add(s);
            return;
        }
        if (open < n) {
            generate(result, open + 1, close, s + "(", n);
        }
        if (close < open) {
            generate(result, open, close + 1, s + "(", n);

        }
    }

    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * 返回表达式的值
     *
     * @param s string字符串 待计算的表达式
     * @return int整型
     */
    public int basicCalculator(String s) {
        // write code here
        if (s == null) {
            return 0;
        }
        s = s.trim();
        if (s.isEmpty()) {
            return 0;
        }
        int result = 0;
        char[] words = s.toCharArray();
        Stack<Integer> stack = new Stack<>();

        // 1 - 2 => +1-2
        char sign = '+';
        int index = 0;
        while (index <= words.length) {
            if (index < words.length) {

                // 如果碰到 1 + (2 - 3 * 5)
                // 递归处理括号内
                if (words[index] == '(') {
                    int lastIndex = getLastIndex(words, index, words.length);
                    String substring = s.substring(index + 1, lastIndex);
                    int val = basicCalculator(substring);
                    stack.push(val);
                    index = lastIndex;
                }
                // 存储整型的数字
                if (Character.isDigit(words[index])) {
                    int tmp = 0;
                    while (index < words.length && Character.isDigit(words[index])) {
                        tmp = tmp * 10 + Character.getNumericValue(words[index]);
                        index++;
                    }
                    stack.push(tmp);
                }
            }
            // 达到字符串末尾 或者达到下一个 符号位的时候 处理前面的表达式
            // 1 - 2 * 3  达到-符号位时
            if (index == words.length || ((words[index] != '(' && words[index] != ')') && words[index] != ' ')) {
                if (sign == '+') {
                    stack.push(stack.pop());
                } else if (sign == '-') {
                    stack.push(-stack.pop());
                } else if (sign == '*') {
                    stack.push(stack.pop() * stack.pop());
                } else if (sign == '/') {
                    int dividend = stack.pop();
                    int divisor = stack.pop();
                    stack.push(divisor / dividend);

                }
                if (index != words.length) {
                    sign = words[index];
                }
            }
            index++;
        }
        for (Integer tmp : stack) {
            result += tmp;
        }
        return result;
    }

    private int getLastIndex(char[] words, int index, int len) {
        int count = 0;
        while (index < len) {
            if (words[index] == '(') {
                count++;
            }
            if (words[index] == ')') {
                count--;
            }
            if (count == 0) {
                return index;
            }
            index++;
        }
        return -1;
    }


    /**
     * 227. Basic Calculator II
     *
     * @param s
     * @return
     */
    public int calculateII(String s) {
        if (s == null) {
            return 0;
        }
        s = s.trim();
        if (s.isEmpty()) {
            return 0;
        }
        char[] words = s.toCharArray();
        int result = 0;
        char sign = '+';
        int endIndex = 0;
        Stack<Integer> stack = new Stack<>();
        while (endIndex < words.length) {
            if (Character.isDigit(words[endIndex])) {
                int tmp = 0;
                while (endIndex < words.length && Character.isDigit(words[endIndex])) {
                    tmp = tmp * 10 + Character.getNumericValue(words[endIndex]);
                    endIndex++;
                }
                stack.push(tmp);
            }
            if (endIndex == words.length || words[endIndex] != ' ') {
                if (sign == '+') {
                    stack.push(stack.pop());
                } else if (sign == '-') {
                    stack.push(stack.pop() * -1);
                } else if (sign == '*') {
                    stack.push(stack.pop() * stack.pop());
                } else if (sign == '/') {
                    int dividend = stack.pop();
                    int divisor = stack.pop();
                    stack.push(divisor / dividend);
                }
                if (endIndex != words.length && words[endIndex] != ' ') {
                    sign = words[endIndex];
                }
            }
            endIndex++;
        }
        for (Integer tmp : stack) {
            result += tmp;
        }
        return result;
    }


    public int calculate(String s) {
        if (s == null) {
            return 0;
        }
        if (s.isEmpty()) {
            return 0;
        }
        char[] words = s.toCharArray();
        int result = 0;
        int endIndex = 0;
        int sign = 1;
        Stack<Integer> stack = new Stack<>();
        while (endIndex < words.length) {
            if (Character.isDigit(words[endIndex])) {
                int tmp = 0;
                while (endIndex < words.length && Character.isDigit(words[endIndex])) {
                    tmp = tmp * 10 + Character.getNumericValue(words[endIndex]);
                    endIndex++;
                }
                result += sign * tmp;
            }
            if (endIndex == words.length || words[endIndex] != ' ') {
                if (endIndex != words.length) {
                    char currentSign = words[endIndex];
                    if (currentSign == '+') {
                        sign = 1;
                    } else if (currentSign == '-') {
                        sign = -1;
                    }
                    if (currentSign == '(') {
                        stack.push(result);
                        stack.push(sign);
                        result = 0;
                        sign = 1;
                    }
                    if (currentSign == ')') {
                        Integer pop = stack.pop();
                        result = pop * result + stack.pop();
                    }
                }
            }
            endIndex++;
        }
        return result;
    }


    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * 返回表达式的值
     *
     * @param s string字符串 待计算的表达式
     * @return int整型
     */
    public int calculatorII(String s) {
        // write code here
        if (s == null) {
            return 0;
        }
        s = s.trim();
        if (s.isEmpty()) {
            return 0;
        }
        int index = 0;
        int len = s.length();
        char[] words = s.toCharArray();
        char sign = '+';
        Stack<Integer> stack = new Stack<>();
        while (index <= len) {
            if (index < len && words[index] == '(') {
                int lastIndexOf = getLastIndex(words, index, words.length);
                int tmp = calculatorII(s.substring(index + 1, lastIndexOf));
                stack.push(tmp);
                index = lastIndexOf;
            }
            if (index < len && Character.isDigit(words[index])) {
                int tmp = 0;
                while (index < len && Character.isDigit(words[index])) {
                    tmp = tmp * 10 + Character.getNumericValue(words[index]);
                    index++;
                }
                stack.push(tmp);
            }
            if (index == len || !Character.isDigit(words[index])) {
                if (sign == '-') {
                    stack.push(-1 * stack.pop());
                } else if (sign == '*') {
                    stack.push(stack.pop() * stack.pop());
                } else if (sign == '/') {
                    Integer second = stack.pop();
                    Integer first = stack.pop();
                    stack.push(first / second);
                }
                if (index != len) {
                    sign = words[index];
                }
            }
            index++;
        }
        int result = 0;
        for (Integer num : stack) {
            result += num;
        }
        return result;
    }

    /**
     * NC96 判断一个链表是否为回文结构
     *
     * @param head ListNode类 the head
     * @return bool布尔型
     */
    public boolean isPail(ListNode head) {
        // write code here
        if (head == null || head.next == null) {
            return true;
        }
        ListNode slow = head;
        ListNode fast = head;
        while (fast.next != null && fast.next.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode next = slow.next;
        slow.next = null;
        ListNode reverse = reverse(next);
        while (head != null && reverse.next != null) {
            if (head.val != reverse.val) {
                return false;
            }
            head = head.next;
            reverse = reverse.next;
        }
        return true;
    }


    private ListNode reverse(ListNode head) {
        ListNode prev = null;
        while (head != null) {
            ListNode next = head.next;
            head.next = prev;
            prev = head;
            head = next;
        }
        return prev;
    }

    /**
     * NC34 求路径
     *
     * @param m int整型
     * @param n int整型
     * @return int整型
     */
    public int uniquePaths(int m, int n) {
        // write code here
        int[] dp = new int[n];
        Arrays.fill(dp, 1);
        for (int i = 1; i < m; i++) {
            for (int j = 0; j < n; j++) {
                dp[j] = dp[j] + (j == 0 ? 0 : dp[j - 1]);
            }
        }
        return dp[n - 1];
    }


    /**
     * todo
     * NC116 把数字翻译成字符串
     *
     * @param nums string字符串 数字串
     * @return int整型
     */
    public int decode(String nums) {
        // write code here
        if (nums == null || nums.isEmpty()) {
            return 0;
        }
        int len = nums.length();

        int[] dp = new int[len + 1];

        dp[0] = 1;

        dp[1] = nums.charAt(0) == '0' ? 0 : 1;

        for (int i = 2; i <= len; i++) {
            int num1 = Integer.parseInt(nums.substring(i - 1, i));
            int num2 = Integer.parseInt(nums.substring(i - 2, i));
            if (num1 >= 1 && num1 <= 9) {
                dp[i] += dp[i - 1];
            }
            if (num2 >= 10 && num2 <= 26) {
                dp[i] += dp[i - 2];
            }
        }
        return dp[len];
    }
}
