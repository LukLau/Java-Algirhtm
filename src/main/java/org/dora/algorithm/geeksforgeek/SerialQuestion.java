package org.dora.algorithm.geeksforgeek;

import org.dora.algorithm.datastructe.ListNode;
import org.dora.algorithm.datastructe.TreeNode;

import java.util.*;

/**
 * @author dora
 * @date 2019/11/20
 */
public class SerialQuestion {


    //    --- 卖股票 ----- //

    public static void main(String[] args) {
        SerialQuestion serialQuestion = new SerialQuestion();
        int result = serialQuestion.lengthOfLongestSubstringTwoDistinct("ababacccccc");
        System.out.println(result);
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
     * key case: 第K次卖出的时候
     * 188. Best Time to Buy and Sell Stock IV
     *
     * @param k
     * @param prices
     * @return
     */
    public int maxProfitIIII(int k, int[] prices) {
        if (prices == null || prices.length == 0) {
            return 0;
        }
        if (k > prices.length / 2) {
            return quickProfit(prices);
        }
        int[][] dp = new int[k + 1][prices.length];
        for (int i = 1; i <= k; i++) {
            int tmp = -prices[0];

            for (int j = 1; j < prices.length; j++) {
                dp[i][j] = Math.max(dp[i][j - 1], prices[j] + tmp);

                tmp = Math.max(tmp, dp[i - 1][j - 1] - prices[j]);
            }
        }
        return dp[k][prices.length - 1];
    }

    private int quickProfit(int[] prices) {
        int result = 0;
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > prices[i - 1]) {
                result += prices[i] - prices[i - 1];
            }
        }
        return result;
    }


    // ------字符串切割-- //

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


    // -------树的遍历---//

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

    // ----树的排序 //

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


    // Olog(N)复杂度 //

    // ------旋转数组 找最小值-----//

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
     * 162. Find Peak Element
     *
     * @param nums
     * @return
     */
    public int findPeakElement(int[] nums) {
        if (nums == null || nums.length == 0) {
            return -1;
        }
        int left = 0;
        int right = nums.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < nums[mid + 1]) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return left;
    }


    // --两个数之和--- //

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


    // ----滑动窗口系列------- //
    // 滑动窗口固定解题公式-- //

    /**
     * 167. Two Sum II - Input array is sorted
     *
     * @param numbers
     * @param target
     * @return
     */
    public int[] twoSum(int[] numbers, int target) {
        if (numbers == null || numbers.length == 0) {
            return new int[]{};
        }
        int[] ans = new int[2];
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < numbers.length; i++) {
            if (map.containsKey(target - numbers[i])) {
                int first = map.get(target - numbers[i]) + 1;
                int second = i + 1;
                ans[0] = first;
                ans[1] = second;
                return ans;
            }
            map.put(numbers[i], i);
        }
        return ans;
    }

    /**
     * 159 Longest Substring with At Most Two Distinct Characters
     *
     * @param s: a string
     * @return: the length of the longest substring T that contains at most 2 distinct characters
     */
    public int lengthOfLongestSubstringTwoDistinct(String s) {
        // Write your code here
        if (s == null || s.isEmpty()) {
            return 0;
        }
        HashMap<Character, Integer> map = new HashMap<>();
        int left = 0;
        int result = 0;
        for (int i = 0; i < s.length(); i++) {
            char localCharacter = s.charAt(i);
            Integer num = map.getOrDefault(localCharacter, 0);
            num++;
            map.put(localCharacter, num);
            while (map.size() > 2) {
                char leftCharacter = s.charAt(left);

                Integer integer = map.get(leftCharacter);

                integer--;

                map.put(leftCharacter, integer);
                if (integer == 0) {
                    map.remove(leftCharacter);
                }
                left++;
            }
            result = Math.max(result, i - left + 1);
        }
        return result;
    }


    // ---字符串反转问题--//

    /**
     * 186 Reverse Words in a String II
     * 不能使用
     *
     * @param str: a string
     * @return: return a string
     */
    public char[] reverseWords(char[] str) {
        // write your code here
        if (str == null || str.length == 0) {
            return new char[]{};
        }
        return null;
    }

    // ----房屋抢劫----//

    /**
     * 198. House Robber
     *
     * @param nums
     * @return
     */
    public int rob(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int column = nums.length;

        int[] dp = new int[column];

        dp[0] = nums[0];

        for (int i = 1; i < nums.length; i++) {
            if (i == 1) {
                dp[i] = Math.max(0, nums[1]);
            } else {
                dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i]);
            }

        }
        return dp[column - 1];

    }


}