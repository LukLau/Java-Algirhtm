package org.dora.algorithm.question;

import org.dora.algorithm.datastructe.ListNode;

import java.util.*;

/**
 * @author dora
 * @date 2019/10/23
 */
public class Question {

    public static void main(String[] args) {
        int[] nums1 = new int[]{1};
        int[] nums2 = new int[]{2, 3};
        Question question = new Question();
        question.findMedianSortedArrays(nums1, nums2);
    }

    /**
     * 4. Median of Two Sorted Arrays
     *
     * @param nums1
     * @param nums2
     * @return
     */
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int m = nums1.length;
        int n = nums2.length;
        if (n < m) {
            return this.findMedianSortedArrays(nums2, nums1);
        }
        int imin = 0;
        int imax = m;
        int maxLeft = 0;
        int minRight = 0;

        boolean odd = ((m + n) % 2 == 1);
        while (imin <= imax) {
            int i = (imin + imax) / 2;
            int j = (m + n) / 2 - i;
            if (i < m && nums1[i] < nums2[j - 1]) {
                imin = i + 1;
            } else if (i > 0 && nums1[i - 1] > nums2[j]) {
                imax = i - 1;
            } else {
                if (i == 0) {
                    maxLeft = nums2[j - 1];
                } else if (i == m) {
                    maxLeft = nums1[i - 1];
                } else {
                    maxLeft = Math.max(nums1[i - 1], nums2[j - 1]);
                }

                if (odd) {
                    return maxLeft;
                }
                if (i == m) {
                    minRight = nums2[j];
                } else if (j == n) {
                    minRight = nums1[i];
                } else {
                    minRight = Math.min(nums1[i], nums2[j]);
                }
                return (maxLeft + minRight) / 2.0;
            }
        }
        return -1;
    }

    /**
     * 6. ZigZag Conversion
     *
     * @param s
     * @return
     */
    public String longestPalindrome(String s) {
        if (s == null || s.isEmpty()) {
            return "";
        }
        int n = s.length();
        int begin = 0;
        int result = 0;
        boolean[][] dp = new boolean[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                if (s.charAt(j) == s.charAt(i)) {
                    if (i - j < 2) {
                        dp[j][i] = true;
                    } else {
                        dp[j][i] = dp[j + 1][i - 1];
                    }
                }
                if (dp[j][i] && i - j + 1 > result) {

                    begin = j;

                    result = i - j + 1;
                }
            }
        }
        if (result != 0) {
            return s.substring(begin, begin + result);
        }
        return s;
    }

    /**
     * @param s
     * @param numRows
     * @return
     */
    public String convert(String s, int numRows) {
        if (s == null || s.isEmpty()) {
            return "";
        }
        StringBuilder[] builders = new StringBuilder[numRows];
        for (int i = 0; i < builders.length; i++) {
            builders[i] = new StringBuilder();
        }
        char[] chars = s.toCharArray();
        int index = 0;
        while (index < chars.length) {
            for (int i = 0; i < numRows && index < chars.length; i++) {
                builders[i].append(chars[index++]);
            }
            for (int i = numRows - 2; i >= 1 && index < chars.length; i--) {
                builders[i].append(chars[index++]);
            }
        }
        for (int i = 1; i < numRows; i++) {
            builders[0].append(builders[i]);
        }
        return builders[0].toString();
    }

    /**
     * 8. String to Integer (atoi)
     *
     * @param str
     * @return
     */
    public int myAtoi(String str) {
        if (str == null) {
            return 0;
        }
        str = str.trim();
        if (str.isEmpty()) {
            return 0;
        }
        int sign = 1;
        int index = 0;
        if (str.charAt(index) == '-' || str.charAt(index) == '+') {
            sign = str.charAt(index) == '-' ? -1 : 1;
            index++;
        }
        long result = 0;
        while (index < str.length() && Character.isDigit(str.charAt(index))) {
            int value = Character.getNumericValue(str.charAt(index));

            result = result * 10 + value;

            index++;

            if (result > Integer.MAX_VALUE) {
                return sign == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
            }
        }
        return (int) (sign * result);
    }

    /**
     * key case: 集合是任何集合的子集
     * 10. Regular Expression Matching
     *
     * @param s
     * @param p
     * @return
     */
    public boolean isMatch(String s, String p) {
        if (s == null) {
            return true;
        }
        if (p == null) {
            return false;
        }
        int m = s.length();
        int n = p.length();
        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[0][0] = true;
        for (int j = 1; j <= n; j++) {
            dp[0][j] = s.charAt(j - 1) == '*' && dp[0][j - 2];
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (s.charAt(i - 1) == p.charAt(j - 1) || p.charAt(j - 1) == '.') {
                    dp[i][j] = dp[i - 1][j - 1];
                }
                if (p.charAt(j - 1) == '*') {
                    if (s.charAt(i - 1) != p.charAt(j - 2) && p.charAt(j - 2) != '.') {
                        dp[i][j] = dp[i][j - 2];
                    } else {
                        dp[i][j] = dp[i][j - 2] || dp[i][j - 1] || dp[i - 1][j];
                    }
                }
            }
        }
        return dp[m][n];
    }

    /**
     * 12. Integer to Roman
     *
     * @param num
     * @return
     */
    public String intToRoman(int num) {
        if (num <= 0) {
            return "";
        }
        String[] bit = new String[]{"", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"};
        String[] ten = new String[]{"", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"};
        String[] hundred = new String[]{"", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"};
        String[] thousand = new String[]{"", "M", "MM", "MMM"};
        StringBuilder builder = new StringBuilder();
        builder.append(thousand[num / 1000]);
        builder.append(hundred[(num % 1000) / 100]);
        builder.append(ten[(num % 100) / 10]);
        builder.append(bit[num % 10]);
        return builder.toString();
    }


    /**
     * 13. Roman to Integer
     *
     * @param s
     * @return
     */
    public int romanToInt(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        }
        int[] ans = new int[s.length()];
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == 'I') {
                ans[i] = 1;
            }
            if (s.charAt(i) == 'V') {
                ans[i] = 5;
            }
            if (s.charAt(i) == 'X') {
                ans[i] = 10;
            }
            if (s.charAt(i) == 'L') {
                ans[i] = 50;
            }
            if (s.charAt(i) == 'C') {
                ans[i] = 100;
            }
            if (s.charAt(i) == 'D') {
                ans[i] = 500;
            }
            if (s.charAt(i) == 'M') {
                ans[i] = 1000;
            }
        }
        int result = 0;
        for (int i = 0; i < ans.length; i++) {
            result += ans[i];
            if (i > 0 && ans[i] > ans[i - 1]) {
                result -= 2 * ans[i - 1];
            }
        }
        return result;
    }


    /**
     * 17. Letter Combinations of a Phone Number
     *
     * @param digits
     * @return
     */
    public List<String> letterCombinations(String digits) {
        if (digits == null || digits.isEmpty()) {
            return new ArrayList<>();
        }
        String[] map = new String[]{"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        LinkedList<String> ans = new LinkedList<>();
        ans.add("");

        return ans;
    }

    /**
     * 23. Merge k Sorted Lists
     *
     * @param lists
     * @return
     */
    public ListNode mergeKLists(ListNode[] lists) {
        if (lists == null || lists.length == 0) {
            return null;
        }
        PriorityQueue<ListNode> priorityQueue = new PriorityQueue<>(lists.length, Comparator.comparingInt(o1 -> o1.val));
        for (ListNode node : lists) {
            if (node != null) {
                priorityQueue.add(node);
            }
        }
        ListNode root = new ListNode(0);
        ListNode dummy = root;
        while (!priorityQueue.isEmpty()) {
            ListNode poll = priorityQueue.poll();
            dummy.next = poll;
            dummy = dummy.next;
            if (poll.next != null) {
                priorityQueue.add(poll.next);
            }
        }
        return root.next;
    }

    /**
     * 24. Swap Nodes in Pairs
     *
     * @param head
     * @return
     */
    public ListNode swapPairs(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode root = new ListNode(0);
        root.next = head;

        ListNode dummy = root;

        while (dummy.next != null && dummy.next.next != null) {

            ListNode slow = dummy.next;

            ListNode fast = dummy.next.next;

            slow.next = fast.next;
            fast.next = slow;
            dummy.next = fast;
            dummy = slow;
        }
        return root.next;
    }


    /**
     * 25. Reverse Nodes in k-Group
     *
     * @param start
     * @param end
     * @return
     */
    public ListNode reverseKGroup(ListNode head, int k) {
        if (head == null || k <= 0) {
            return head;
        }
        ListNode current = head;
        for (int i = 0; i < k; i++) {
            if (current == null) {
                return head;
            }
            current = current.next;
        }
        ListNode root = this.reverseList(head, current);
        head.next = this.reverseKGroup(current, k);
        return root;
    }

    private ListNode reverseList(ListNode start, ListNode end) {
        ListNode prev = end;
        while (start != end) {
            ListNode tmp = start.next;

            start.next = prev;

            prev = start;

            start = tmp;
        }
        return prev;
    }

    /**
     * todo KMP 算法
     * 28. Implement strStr()
     *
     * @param haystack
     * @param needle
     * @return
     */
    public int strStr(String haystack, String needle) {
        if (haystack == null || needle == null) {
            return -1;
        }
        int m = haystack.length();

        int n = needle.length();

        for (int i = 0; i <= m - n; i++) {

            int j = 0;

            while (j < n && haystack.charAt(i + j) == needle.charAt(j)) {
                j++;
            }
            if (j == n) {
                return i;
            }
        }
        return -1;
    }
}

