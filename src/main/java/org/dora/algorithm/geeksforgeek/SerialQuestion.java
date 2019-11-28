package org.dora.algorithm.geeksforgeek;

import org.dora.algorithm.datastructe.ListNode;
import org.dora.algorithm.datastructe.TreeNode;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Stack;

/**
 * @author dora
 * @date 2019/11/20
 */
public class SerialQuestion {


    //    --- 卖股票 ----- //

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
        int minPrice = prices[0];
        int result = 0;
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > minPrice) {
                result = Math.max(result, prices[i] - minPrice);
            } else {
                minPrice = prices[i];
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
            if (prices[i] > minPrice) {
                result += prices[i] - minPrice;
            }
            minPrice = prices[i];
        }
        return result;
    }

    /**
     * todo 股票交易两次
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

        int minLeftPrice = prices[0];

        int leftResult = 0;

        for (int i = 1; i < n; i++) {
            if (prices[i] < minLeftPrice) {
                minLeftPrice = prices[i];
            }

            leftResult = Math.max(leftResult, prices[i] - minLeftPrice);

            leftProfit[i] = leftResult;
        }
        int[] rightProfit = new int[n + 1];

        int maxRightPrice = prices[n - 1];

        int rightResult = 0;

        for (int i = n - 2; i >= 1; i--) {
            if (maxRightPrice < prices[i]) {

                maxRightPrice = prices[i];
            }

            rightResult = Math.max(rightResult, maxRightPrice - prices[i]);

            rightProfit[i] = rightResult;

        }

        int result = 0;

        for (int i = 0; i < n; i++) {
            result = Math.max(result, leftProfit[i] + rightProfit[i + 1]);
        }
        return result;


    }


    // ------字符串切割-- //

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
            if (s.charAt(start) == s.charAt(end)) {
                start++;
                end--;
            } else {
                return false;
            }
        }
        return true;
    }


    /**
     * cut[i] is the minimum of cut[j - 1] + 1 (j <= i), if [j, i] is palindrome.
     * If [j, i] is palindrome, [j + 1, i - 1] is palindrome, and c[j] == c[i].
     * <p>
     * todo
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
        int[] cut = new int[n];

        boolean[][] dp = new boolean[n][n];
        for (int i = 0; i < n; i++) {
            int cutNum = i;
            for (int j = 0; j <= i; j++) {
                if (s.charAt(j) == s.charAt(i) && (i - j < 2 || dp[j + 1][i - 1])) {
                    dp[j][i] = true;
                    cutNum = j == 0 ? 0 : Math.min(cutNum, cut[j - 1] + 1);
                }
            }
            cut[i] = cutNum;
        }
        return cut[n - 1];


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


    /**
     * 142. Linked List Cycle II
     *
     * @param head
     * @return
     */
    public ListNode detectCycle(ListNode head) {
        if (head == null || head.next == null) {
            return null;
        }
        ListNode fast = head;
        ListNode slow = head;
        while (fast.next != null && fast.next.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast) {
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


    // -------树的遍历---//

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
        TreeNode p = root;
        Stack<TreeNode> stack = new Stack<>();
        while (!stack.isEmpty() || p != null) {
            if (p != null) {
                stack.push(p);
                ans.addFirst(p.val);
                p = p.right;
            } else {
                p = stack.pop();
                p = p.left;
            }
        }
        return ans;
    }

    // ----树的排序 //

    /**
     * 插入排序
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
        ListNode node = slow.next;
        slow.next = null;
        ListNode l1 = this.sortList(head);
        ListNode l2 = this.sortList(node);
        return this.sortListNode(l1, l2);

    }

    private ListNode sortListNode(ListNode l1, ListNode l2) {
        if (l1 == null) {
            return l2;
        }
        if (l2 == null) {
            return l1;
        }
        if (l1.val < l2.val) {
            l1.next = this.sortListNode(l1.next, l2);
            return l1;
        } else {
            l2.next = this.sortListNode(l1, l2.next);
            return l2;
        }
    }


    // Olog(N)复杂度 //

    // ------旋转数组 找最小值-----//

    /**
     * key point: 左边界可能存在两个区域 所以无法很好的做边界条件。
     * 故使用右边界
     * 153. Find Minimum in Rotated Sorted Array
     *
     * @param nums
     * @return
     */
    public int findMin(int[] nums) {
        if (nums == null || nums.length == 0) {
            return -1;
        }
        int left = 0;

        int right = nums.length - 1;

        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] <= nums[right]) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return nums[left];
    }

    /**
     * todo 未来需要三种
     * 154. Find Minimum in Rotated Sorted Array II
     *
     * @param nums
     * @return
     */
    public int findMinII(int[] nums) {
        if (nums == null || nums.length == 0) {
            return -1;
        }
        int left = 0;
        int right = nums.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == nums[right]) {
                right--;
            } else if (nums[mid] < nums[right]) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return nums[left];

    }


    // -------- 逆波兰数---//

    /**
     * 150. Evaluate Reverse Polish Notation
     *
     * @param tokens
     * @return
     */
    public int evalRPN(String[] tokens) {
        if (tokens == null || tokens.length == 0) {
            return 0;
        }
        Stack<Integer> stack = new Stack<>();
        for (String token : tokens) {
            if (token.equals("+")) {
                int second = stack.pop();
                int first = stack.pop();
                stack.push(first + second);
            } else if (token.equals("-")) {
                int second = stack.pop();
                int first = stack.pop();
                stack.push(first - second);

            } else if (token.equals("*")) {
                int second = stack.pop();
                int first = stack.pop();
                stack.push(first * second);
            } else if (token.equals("/")) {
                int second = stack.pop();
                int first = stack.pop();
                stack.push(first / second);
            } else {
                stack.push(Integer.parseInt(token));
            }
        }
        return stack.pop();
    }


}
