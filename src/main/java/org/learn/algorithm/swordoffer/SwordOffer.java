package org.learn.algorithm.swordoffer;

import jdk.internal.org.objectweb.asm.tree.MultiANewArrayInsnNode;
import org.learn.algorithm.datastructure.Interval;
import org.learn.algorithm.datastructure.ListNode;
import org.learn.algorithm.datastructure.TreeNode;

import javax.print.attribute.standard.NumberOfInterveningJobs;
import javax.sound.midi.Soundbank;
import java.util.*;

/**
 * 剑指offer
 *
 * @author luk
 * @date 2021/5/2
 */
public class SwordOffer {

    public static void main(String[] args) {
        SwordOffer offer = new SwordOffer();
        int[] ans = new int[]{2, 1, 5, 3, 6, 4, 8, 9, 7};
        int[] lis = offer.LIS(ans);
        for (int li : lis) {
            System.out.println(li);
        }
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
     * longest common sequence
     *
     * @param s1 string字符串 the string
     * @param s2 string字符串 the string
     * @return string字符串
     */
    public String LCSV2(String s1, String s2) {
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
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        StringBuilder builder = new StringBuilder();
        int k = m;
        int j = n;
        while (k > 0 && j > 0) {
            if (s1.charAt(k - 1) == s2.charAt(j - 1)) {
                builder.append(s1.charAt(k - 1));
                k--;
                j--;
            } else {
                if (dp[k - 1][j] > dp[k][j - 1]) {
                    k--;
                } else {
                    j--;
                }
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
        int[][] dp = new int[m + 1][n + 1];
        int max = 0;
        int index = 0;
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (str1.charAt(i - 1) == str2.charAt(j - 1)) {
                    dp[i][j] = 1 + dp[i - 1][j - 1];
                    if (dp[i][j] > max) {
                        max = dp[i][j];
                        index = i;
                    }
                }

            }
        }
        if (max == 0) {
            return "";
        }
        return str1.substring(index - max + 1, index);
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
            dp[i][0] = ic + dp[i - 1][0];
        }
        for (int j = 1; j <= n; j++) {
            dp[0][j] = ic + dp[0][j - 1];
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (str1.charAt(i - 1) == str2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = Math.min(dp[i - 1][j] + dc, Math.min(dp[i][j - 1] + ic, dp[i - 1][j - 1] + rc));
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
        if (arr.length == 1) {
            return arr;
        }
        int len = arr.length;
        int[] dp = new int[len];
        Arrays.fill(dp, 1);
        int max = 0;
        int endIndex = -1;
        for (int i = 1; i < arr.length; i++) {
            for (int j = 0; j < i; j++) {
                if (arr[j] < arr[i] && dp[i] < dp[j] + 1) {
                    dp[i] = dp[j] + 1;
                }
            }
            if (dp[i] > max) {
                max = dp[i];
                endIndex = i;
            }
        }
        int m = max;
        int[] pos = new int[m];
        pos[m - 1] = arr[endIndex];
        int index = 1;
        int k = endIndex - 1;
        while (k >= 1) {
            if (dp[k] == max - 1) {
                pos[m - 1 - index] = arr[k];
                max--;
                index++;
            }
            k--;
        }
        return pos;
    }


    /**
     * @param intervals
     * @return
     */
    public ArrayList<Interval> merge(ArrayList<Interval> intervals) {
        if (intervals == null || intervals.isEmpty()) {
            return new ArrayList<>();
        }
        intervals.sort(Comparator.comparingInt(o -> o.start));
        LinkedList<Interval> result = new LinkedList<>();
        int size = intervals.size();
        for (int i = 0; i < size; i++) {
            Interval interval = intervals.get(i);
            if (result.isEmpty() || result.peekLast().end < interval.start) {
                result.add(interval);
            } else {
                Interval pre = result.peekLast();
                pre.start = Math.min(pre.start, interval.start);
                pre.end = Math.max(pre.end, interval.end);
            }
        }
        return new ArrayList<>(result);
    }


    /**
     * @param str string字符串
     * @return int整型
     */
    public int atoi(String str) {
        // write code here
        if (str == null || str.isEmpty()) {
            return 0;
        }
        str = str.trim();
        char[] words = str.toCharArray();
        int sign = 1;
        int index = 0;
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

    private int depth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        return 1 + Math.max(depth(root.left), depth(root.right));
    }


    /**
     * @param prices int整型一维数组
     * @return int整型
     */
    public int maxProfit(int[] prices) {
        // write code here
        if (prices == null || prices.length == 0) {
            return 0;
        }
        int result = 0;
        int cost = prices[0];
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > cost) {
                result = Math.max(result, prices[i] - cost);
            } else {
                cost = prices[i];
            }
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
     *
     * @param root TreeNode类
     * @param o1 int整型
     * @param o2 int整型
     * @return int整型
     */
    public int lowestCommonAncestor (TreeNode root, int o1, int o2) {
        // write code here
        return -1;
    }



}
