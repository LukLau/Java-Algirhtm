package org.dora.algorithm.solution.v1;


import org.dora.algorithm.datastructe.Interval;
import org.dora.algorithm.datastructe.ListNode;
import org.dora.algorithm.datastructe.TreeNode;

import java.util.*;

/**
 * @author liulu
 * @date 2019/02/16
 */
@Deprecated
public class OneHundred {


    public static void main(String[] args) {

    }

    /**
     * 1. Two Sum
     *
     * @param nums
     * @param target
     * @return
     */
    public int[] twoSum(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return new int[2];
        }
        int[] ans = new int[2];
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(target - nums[i])) {
                ans[0] = map.get(target - nums[i]);
                ans[1] = i;
                return ans;
            }
            map.put(nums[i], i);
        }
        return ans;
    }

    /**
     * 2. Add Two Numbers
     *
     * @param l1
     * @param l2
     * @return
     */
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        if (l1 == null && l2 == null) {
            return null;
        }
        int carry = 0;
        ListNode root = new ListNode(0);
        ListNode dummy = root;
        while (l1 != null || l2 != null || carry != 0) {
            int sum = (l1 == null ? 0 : l1.val) + (l2 == null ? 0 : l2.val) + carry;

            carry = sum / 10;

            ListNode node = new ListNode(sum % 10);
            dummy.next = node;
            dummy = dummy.next;

            l1 = l1 == null ? null : l1.next;
            l2 = l2 == null ? null : l2.next;
        }
        return root.next;
    }

    /**
     * 3. Longest Substring Without Repeating Characters
     *
     * @param s
     * @return
     */
    public int lengthOfLongestSubstring(String s) {
        if (s == null || s.length() == 0) {
            return 0;
        }
        int left = 0;
        int longest = 0;
        HashMap<Character, Integer> map = new HashMap<>();
        for (int i = 0; i < s.length(); i++) {
            if (map.containsKey(s.charAt(i))) {
                left = Math.max(left, map.get(s.charAt(i)) + 1);
            }
            longest = Math.max(longest, i - left + 1);
            map.put(s.charAt(i), i);
        }
        return longest;
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
        if (m > n) {
            return this.findMedianSortedArrays(nums2, nums1);
        }
        int imin = 0, imax = m, max_left = 0, min_right = 0;
        while (imin <= imax) {
            int i = (imin + imax) / 2;
            int j = (m + n + 1) / 2 - i;
            if (i > 0 && nums1[i - 1] > nums2[j]) {
                imax = i - 1;
            } else if (i < m && nums1[i] < nums2[j - 1]) {
                imin = i + 1;
            } else {
                if (i == m) {
                    max_left = nums1[i - 1];
                } else if (i == 0) {
                    max_left = nums2[j - 1];
                }
            }
        }
        return -1;

    }

    /**
     * 5. Longest Palindromic Substring
     *
     * @param s
     * @return
     */
    public String longestPalindrome(String s) {
        if (s == null || s.length() == 0) {
            return "";
        }
        int m = s.length();
        int left = 0;
        int longest = 0;
        boolean[][] dp = new boolean[m][m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j <= i; j++) {
                if (i - j <= 2) {
                    dp[j][i] = s.charAt(j) == s.charAt(i);
                } else {
                    dp[j][i] = s.charAt(j) == s.charAt(i) && dp[j + 1][i - 1];
                }
                if (dp[j][i] && i - j + 1 > longest) {
                    left = j;
                    longest = i - j + 1;
                }

            }
        }
        if (longest > 0) {
            return s.substring(left, left + longest);
        }
        return s;
    }

    /**
     * 6. ZigZag Conversion
     *
     * @param s
     * @param numRows
     * @return
     */
    public String convert(String s, int numRows) {
        if (s == null || s.length() == 0) {
            return "";
        }
        StringBuilder[] stringBuilders = new StringBuilder[numRows];
        for (int i = 0; i < numRows; i++) {
            stringBuilders[i] = new StringBuilder();
        }
        int idx = 0;
        char[] chars = s.toCharArray();
        while (idx < s.length()) {
            for (int i = 0; i < numRows && idx < chars.length; i++) {
                stringBuilders[i].append(chars[idx++]);
            }
            for (int i = numRows - 2; i >= 1 && idx < chars.length; i--) {
                stringBuilders[i].append(chars[idx++]);
            }
        }
        for (int i = 1; i < numRows; i++) {
            stringBuilders[0].append(stringBuilders[i]);
        }
        return stringBuilders[0].toString();
    }

    /**
     * 7. Reverse Integer
     */
    public int reverse(int x) {
        int sum = 0;
        while (x != 0) {
            if (sum > Integer.MAX_VALUE / 10 || sum < Integer.MIN_VALUE / 10) {
                return 0;
            }
            sum = sum * 10 + x % 10;
            x /= 10;
        }
        return sum;
    }

    /**
     * 8. String to Integer (atoi)
     *
     * @param str
     * @return
     */
    public int myAtoi(String str) {
        if (str == null || str.length() == 0) {
            return 0;
        }
        str = str.trim();
        if (str.length() == 0) {
            return 0;
        }
        int sign = 1;
        int idx = 0;
        if (str.charAt(idx) == '-' || str.charAt(idx) == '+') {
            sign = str.charAt(idx) == '-' ? -1 : 1;
            idx++;
        }
        Long ans = 0L;
        while (idx < str.length() && Character.isDigit(str.charAt(idx))) {
            int value = str.charAt(idx) - '0';
            ans = ans * 10 + value;
            if (ans > Integer.MAX_VALUE) {
                return sign == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
            }
            idx++;
        }
        return sign * ans.intValue();
    }

    /**
     * 9. Palindrome Number
     *
     * @param x
     * @return
     */
    public boolean isPalindrome(int x) {
        if (x == 0) {
            return true;
        }
        if (x % 10 == 0) {
            return false;
        }
        int sum = 0;
        while (x > sum) {
            sum = sum * 10 + x % 10;
            x /= 10;
        }
        return sum == x || sum / 10 == x;
    }

    /**
     * 10. Regular Expression Matching
     *
     * @param s
     * @param p
     * @return
     */
    public boolean isMatch(String s, String p) {
        if (s == null || p == null) {
            return false;
        }
        int m = s.length();
        int n = p.length();
        boolean[][] dp = this.initDp(p, m, n);
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (s.charAt(i - 1) == p.charAt(j - 1) || p.charAt(j - 1) == '.') {
                    dp[i][j] = dp[i - 1][j - 1];
                }
                if (p.charAt(j - 1) == '*') {
                    if (s.charAt(i - 1) != p.charAt(j - 2) && p.charAt(j - 2) != '.') {
                        dp[i][j] = dp[i][j - 2];
                    } else {
                        dp[i][j] = dp[i][j - 1] || dp[i][j - 2] || dp[i - 1][j];
                    }
                }
            }
        }
        return dp[m][n];
    }

    /**
     * 11. Container With Most Water
     *
     * @param height
     * @return
     */
    public int maxArea(int[] height) {
        if (height == null || height.length == 0) {
            return 0;
        }
        int result = 0;
        int left = 0;
        int right = height.length - 1;
        while (left < right) {
            result = Math.max(result, Math.min(height[left], height[right]) * (right - left));
            if (height[left] <= height[right]) {
                left++;
            } else {
                right--;
            }
        }
        return result;
    }

    /**
     * 14. Longest Common Prefix
     *
     * @param strs
     * @return
     */
    public String longestCommonPrefix(String[] strs) {
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
     * 15. 3Sum
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> threeSum(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new ArrayList<>();
        }
        List<List<Integer>> ans = new ArrayList<>();
        Arrays.sort(nums);
        for (int i = 0; i < nums.length - 2; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            int left = i + 1;
            int right = nums.length - 1;
            int target = 0 - nums[i];
            while (left < right) {
                if (nums[left] + nums[right] == target) {
                    ans.add(Arrays.asList(nums[i], nums[left], nums[right]));
                    while (left < right && nums[left] == nums[left + 1]) {
                        left++;
                    }
                    while (left < right && nums[right] == nums[right - 1]) {
                        right--;
                    }
                    left++;
                    right--;
                } else if (nums[left] + nums[right] < target) {
                    left++;
                } else {
                    right--;
                }
            }
        }
        return ans;
    }

    /**
     * 16. 3Sum Closest
     *
     * @param nums
     * @param target
     * @return
     */
    public int threeSumClosest(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return Integer.MAX_VALUE;
        }
        int result = nums[0] + nums[1] + nums[2];
        Arrays.sort(nums);
        for (int i = 0; i < nums.length - 2; i++) {
            int left = i + 1;
            int right = nums.length - 1;
            while (left < right) {
                int sum = nums[i] + nums[left] + nums[right];
                if (sum == target) {
                    return sum;
                } else if (Math.abs(sum - target) < Math.abs(result - target)) {
                    result = sum;
                }
                if (sum < target) {
                    left++;
                } else {
                    right--;
                }
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
        if (digits == null || digits.length() == 0) {
            return new ArrayList<>();
        }
        String[] mp = new String[]{"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        LinkedList<String> ans = new LinkedList<>();
        ans.add("");
        for (int i = 0; i < digits.length(); i++) {
            int index = digits.charAt(i) - '0';
            String value = mp[index];
            while (ans.peek().length() == i) {
                String peek = ans.poll();
                for (char c : value.toCharArray()) {
                    ans.add(peek + c);
                }
            }
        }
        return ans;
    }

    /**
     * 18. 4Sum
     *
     * @param nums
     * @param target
     * @return
     */
    public List<List<Integer>> fourSum(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return new ArrayList<>();
        }
        List<List<Integer>> ans = new ArrayList<>();
        Arrays.sort(nums);
        for (int i = 0; i < nums.length - 3; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            for (int j = i + 1; j < nums.length - 2; j++) {
                if (j > i + 1 && nums[j] == nums[j - 1]) {
                    continue;
                }
                int left = j + 1, right = nums.length - 1;
                while (left < right) {
                    int sum = nums[i] + nums[j] + nums[left] + nums[right];
                    if (sum == target) {
                        ans.add(Arrays.asList(nums[i], nums[j], nums[left], nums[right]));
                        while (left < right && nums[left] == nums[left + 1]) {
                            left++;
                        }
                        while (left < right && nums[right] == nums[right - 1]) {
                            right--;
                        }
                        left++;
                        right--;
                    } else if (sum < target) {
                        left++;
                    } else {
                        right--;
                    }
                }
            }
        }
        return ans;
    }

    /**
     * 19. Remove Nth Node From End of List
     *
     * @param head
     * @param n
     * @return
     */
    public ListNode removeNthFromEnd(ListNode head, int n) {
        if (head == null || n < 0) {
            return head;
        }
        ListNode root = new ListNode(0);
        root.next = head;
        ListNode fast = root;
        ListNode slow = root;
        for (int i = 0; i < n; i++) {
            fast = fast.next;
        }
        while (fast.next != null) {
            slow = slow.next;
            fast = fast.next;
        }
        ListNode node = slow.next;
        slow.next = node.next;
        node.next = null;
        return root.next;
    }

    /**
     * 20. Valid Parentheses
     *
     * @param s
     * @return
     */
    public boolean isValid(String s) {
        if (s == null || s.length() == 0) {
            return true;
        }
        Deque<Character> stack = new LinkedList<>();
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') {
                stack.push(')');
            } else if (s.charAt(i) == '[') {
                stack.push(']');
            } else if (s.charAt(i) == '{') {
                stack.push('}');
            } else {
                if (stack.isEmpty() || stack.peek() != s.charAt(i)) {
                    return false;
                } else {
                    stack.pop();
                }
            }
        }
        return stack.isEmpty();
    }

    /**
     * 21. Merge Two Sorted Lists
     *
     * @param l1
     * @param l2
     * @return
     */
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
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
            l1.next = this.mergeTwoLists(l1.next, l2);
            return l1;
        } else {
            l2.next = this.mergeTwoLists(l1, l2.next);
            return l2;
        }
    }

    /**
     * 22. Generate Parentheses
     *
     * @param n
     * @return
     */
    public List<String> generateParenthesis(int n) {
        if (n <= 0) {
            return new ArrayList<>();
        }
        List<String> ans = new ArrayList<>();
        this.generateParenthesis(ans, "", 0, 0, n);
        return ans;
    }

    private void generateParenthesis(List<String> ans, String s, int open, int close, int n) {
        if (s.length() == 2 * n) {
            ans.add(s);
        }
        if (open < n) {
            this.generateParenthesis(ans, s + "(", open + 1, close, n);
        }
        if (close < open) {
            this.generateParenthesis(ans, s + ")", open, close + 1, n);
        }
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

        PriorityQueue<ListNode> priorityQueue = new PriorityQueue<>(Comparator.comparingInt(o -> o.val));
        for (ListNode node : lists) {
            if (node != null) {
                priorityQueue.add(node);
            }
        }
        ListNode root = new ListNode(0);
        ListNode dummy = root;
        while (!priorityQueue.isEmpty()) {
            ListNode node = priorityQueue.poll();
            dummy.next = node;
            dummy = dummy.next;
            if (node.next != null) {
                priorityQueue.add(node.next);
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
            ListNode fast = dummy.next.next;
            ListNode slow = dummy.next;

            dummy.next = fast;
            slow.next = fast.next;
            fast.next = slow;

            dummy = dummy.next.next;
        }
        return root.next;
    }

    /**
     * 25. Reverse Nodes in k-Group
     *
     * @param head
     * @param k
     * @return
     */
    public ListNode reverseKGroup(ListNode head, int k) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode currNode = head;
        for (int i = 0; i < k; i++) {
            if (currNode == null) {
                return head;
            }
            currNode = currNode.next;
        }
        ListNode newHead = this.reverseList(head, currNode);
        head.next = this.reverseKGroup(currNode, k);
        return newHead;
    }

    private ListNode reverseList(ListNode first, ListNode last) {
        ListNode prev = last;
        while (first != last) {
            ListNode tmp = first.next;
            first.next = prev;
            prev = first;
            first = tmp;
        }
        return prev;
    }

    /**
     * 26. Remove Duplicates from Sorted Array
     *
     * @param nums
     * @return
     */
    public int removeDuplicates(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int idx = 1;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] != nums[i - 1]) {
                nums[idx++] = nums[i];
            }
        }
        return idx;
    }

    /**
     * 29. Divide Two Integers
     *
     * @param dividend
     * @param divisor
     * @return
     */
    public int divide(int dividend, int divisor) {
        if (dividend == Integer.MIN_VALUE && divisor == -1) {
            return Integer.MAX_VALUE;
        }
        int sign = (dividend < 0) ^ (divisor < 0) ? -1 : 1;
        long dvd = Math.abs(dividend);
        long dvs = Math.abs(divisor);
        Long result = 0L;
        while (dvd >= dvs) {
            long multi = 1;
            long tmp = dvs;
            while (dvd >= (tmp << 1)) {
                multi <<= 1;
                tmp <<= 1;
            }
            result += multi;
            dvd -= tmp;
        }
        return sign * result.intValue();
    }

    /**
     * 31. Next Permutation
     *
     * @param nums
     */
    public void nextPermutation(int[] nums) {
        if (nums == null || nums.length == 0) {
            return;
        }
        int index = nums.length - 1;
        while (index > 0 && nums[index] < nums[index - 1]) {
            index--;
        }
        if (index == 0) {
            this.reverseArray(nums, index, nums.length - 1);
        } else {
            int j = nums.length - 1;
            while (j > index - 1) {
                if (nums[j] > nums[index - 1]) {
                    break;
                }
                j--;
            }
            this.swap(nums, index - 1, j);
            this.reverseArray(nums, index, nums.length - 1);
        }
    }

    private void reverseArray(int[] nums, int start, int end) {
        if (start > end) {
            return;
        }
        for (int i = start; i <= (start + end) / 2; i++) {
            this.swap(nums, i, start + end - i);
        }
    }

    private void swap(int[] nums, int i, int j) {
        int tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;
    }

    /**
     * 32. Longest Valid Parentheses
     *
     * @param s
     * @return
     */
    public int longestValidParentheses(String s) {
        if (s == null || s.length() == 0) {
            return 0;
        }
        Deque<Integer> stack = new LinkedList<>();
        int result = 0;
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') {
                stack.push(i);
            } else {
                if (!stack.isEmpty() && s.charAt(stack.peek()) == '(') {
                    stack.pop();
                } else {
                    stack.push(i);
                }
            }
        }
        if (stack.isEmpty()) {
            return s.length();
        }
        int a = s.length();
        while (!stack.isEmpty()) {
            result = Math.max(result, a - 1 - stack.peek());
            a = stack.pop();
        }
        result = Math.max(result, a);
        return result;
    }

    /**
     * 33. Search in Rotated Sorted Array
     *
     * @param nums
     * @param target
     * @return
     */
    public int search(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return -1;
        }
        int left = 0;
        int right = nums.length - 1;
        // todo 待学习
        return -1;
    }

    /**
     * 39. Combination Sum
     *
     * @param candidates
     * @param target
     * @return
     */
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        if (this.isEmpty(candidates)) {
            return new ArrayList<>();
        }
        Arrays.sort(candidates);
        List<List<Integer>> ans = new ArrayList<>();
        this.combinationSum(ans, new ArrayList<>(), 0, candidates, target);
        return ans;
    }

    private void combinationSum(List<List<Integer>> ans, List<Integer> integers, int index, int[] candidates, int target) {
        if (target == 0) {
            ans.add(new ArrayList<>(integers));
        }
        for (int i = index; i < candidates.length && candidates[i] <= target; i++) {
            integers.add(candidates[i]);
            this.combinationSum(ans, integers, i, candidates, target - candidates[i]);
            integers.remove(integers.size() - 1);
        }
    }

    private boolean isEmpty(int[] array) {
        return array == null || array.length == 0;
    }

    /**
     * 40. Combination Sum II
     *
     * @param candidates
     * @param target
     * @return
     */
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        if (this.isEmpty(candidates)) {
            return new ArrayList<>();
        }
        List<List<Integer>> ans = new ArrayList<>();
        Arrays.sort(candidates);
        this.combinationSum2(ans, new ArrayList<>(), 0, candidates, target);
        return ans;
    }

    private void combinationSum2(List<List<Integer>> ans, List<Integer> tmp, int index, int[] candidates, int target) {
        if (target == 0) {
            ans.add(new ArrayList<>(tmp));
        }
        for (int i = index; i < candidates.length && candidates[i] <= target; i++) {
            if (i > index && candidates[i] == candidates[i - 1]) {
                continue;
            }
            tmp.add(candidates[i]);
            this.combinationSum2(ans, tmp, i + 1, candidates, target - candidates[i]);
            tmp.remove(tmp.size() - 1);
        }
    }

    /**
     * 41. First Missing Positive
     *
     * @param nums
     * @return
     */
    public int firstMissingPositive(int[] nums) {
        if (this.isEmpty(nums)) {
            return 1;
        }
        for (int i = 0; i < nums.length; i++) {
            while (nums[i] > 0 && nums[i] <= nums.length && nums[i] != nums[nums[i] - 1]) {
                this.swap(nums, i, nums[i] - 1);
            }
        }
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != i + 1) {
                return i + 1;
            }
        }
        return nums.length + 1;
    }

    /**
     * 42. Trapping Rain Water
     *
     * @param height
     * @return
     */
    public int trap(int[] height) {
        if (this.isEmpty(height)) {
            return 0;
        }
        int result = 0;
        int left = 0;
        int right = height.length - 1;
        while (left < right) {
            while (left < right && height[left] == 0) {
                left++;
            }
            while (left < right && height[right] == 0) {
                right--;
            }
            int min = Math.min(height[left], height[right]);
            for (int i = left; i <= right; i++) {
                if (height[i] >= min) {
                    height[i] -= min;
                } else {
                    result += min - height[i];
                    height[i] = 0;
                }
            }
        }
        return result;
    }

    /**
     * 43. Multiply Strings
     *
     * @param num1
     * @param num2
     * @return
     */
    public String multiply(String num1, String num2) {
        if (this.isStringEmpty(num1) || this.isStringEmpty(num2)) {
            return "";
        }
        int m = num1.length();
        int n = num2.length();
        int[] position = new int[m + n];
        for (int i = m - 1; i >= 0; i--) {
            for (int j = n - 1; j >= 0; j--) {
                int value = (num1.charAt(i) - '0') * (num2.charAt(j) - '0') + position[i + j + 1];
                position[i + j + 1] = value % 10;
                position[i + j] += value / 10;
            }
        }
        StringBuilder sb = new StringBuilder();
        for (int num : position) {
            if (sb.length() == 0 && num == 0) {
                continue;
            }
            sb.append(num);
        }
        return sb.length() == 0 ? "0" : sb.toString();
    }

    private boolean isStringEmpty(String string) {
        return string == null || string.length() == 0;
    }

    /**
     * 44. Wildcard Matching
     *
     * @param s
     * @param p
     * @return
     */
    public boolean isMatchII(String s, String p) {
        if (s == null || p == null) {
            return false;
        }
        int m = s.length();
        int n = p.length();
        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[0][0] = true;
        for (int i = 0; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (p.charAt(j - 1) == '*') {
                    dp[i][j] = dp[i][j - 1] || (i > 0 && dp[i - 1][j]);
                } else {
                    dp[i][j] = this.match(s, p, i, j) ? dp[i - 1][j - 1] : false;
                }
            }
        }
        return dp[m][n];
    }

    private boolean match(String s, String p, int i, int j) {
        if (i == 0) {
            return false;
        } else if (p.charAt(j - 1) == '?') {
            return true;
        } else if (s.charAt(i - 1) == p.charAt(j - 1)) {
            return true;
        }
        return false;
    }

    private boolean[][] initDp(String p, int m, int n) {
        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[0][0] = true;
        for (int j = 1; j <= n; j++) {
            dp[0][j] = p.charAt(j - 1) == '*' && dp[0][j - 2];
        }
        return dp;
    }

    /**
     * 45. Jump Game II
     *
     * @param nums
     * @return
     */
    public int jump(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int step = 0;
        int currEnd = 0;
        int currFarthest = 0;
        for (int i = 0; i < nums.length - 1; i++) {
            currFarthest = Math.max(currFarthest, i + nums[i]);
            if (i == currEnd) {
                currEnd = currFarthest;
                step++;
            }
        }
        return step;
    }

    /**
     * 46. Permutations
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> permute(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new ArrayList<>();
        }
        List<List<Integer>> ans = new ArrayList<>();
        boolean[] used = new boolean[nums.length];
        this.permute(ans, new ArrayList<>(), used, nums);
        return ans;
    }

    private void permute(List<List<Integer>> ans, List<Integer> tmp, boolean[] used, int[] nums) {
        if (tmp.size() == nums.length) {
            ans.add(new ArrayList<>(tmp));
        }
        for (int i = 0; i < nums.length; i++) {
            if (used[i]) {
                continue;
            }
            used[i] = true;
            tmp.add(nums[i]);
            this.permute(ans, tmp, used, nums);
            used[i] = false;
            tmp.remove(tmp.size() - 1);
        }
    }

    /**
     * 47. Permutations II
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> permuteUnique(int[] nums) {
        if (this.isEmpty(nums)) {
            return new ArrayList<>();
        }
        Arrays.sort(nums);
        List<List<Integer>> ans = new ArrayList<>();
        boolean[] used = new boolean[nums.length];
        this.permuteUnique(ans, new ArrayList<>(), used, nums);
        return ans;
    }

    private void permuteUnique(List<List<Integer>> ans, List<Integer> tmp, boolean[] used, int[] nums) {
        if (tmp.size() == nums.length) {
            ans.add(new ArrayList<>(tmp));
        }
        for (int i = 0; i < nums.length; i++) {
            if (used[i] || (i > 0 && nums[i] == nums[i - 1] && !used[i - 1])) {
                continue;
            }
            tmp.add(nums[i]);
            used[i] = true;
            this.permuteUnique(ans, tmp, used, nums);
            tmp.remove(tmp.size() - 1);
            used[i] = false;
        }
    }

    /**
     * 48. Rotate Image
     *
     * @param matrix
     */
    public void rotate(int[][] matrix) {
        if (matrix == null || matrix.length == 0) {
            return;
        }

        for (int i = 0; i < matrix.length; i++) {
            for (int j = i; j < matrix[i].length; j++) {
                this.swapMatrix(matrix, i, j);
            }
        }
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length / 2; j++) {
                this.swapMatrixRow(matrix, i, j);
            }
        }
    }

    private void swapMatrixRow(int[][] matrix, int i, int j) {
        int tmp = matrix[i][j];
        matrix[i][j] = matrix[i][matrix[i].length - 1 - j];
        matrix[i][matrix[i].length - 1 - j] = tmp;
    }

    private void swapMatrix(int[][] matrix, int i, int j) {
        int tmp = matrix[i][j];
        matrix[i][j] = matrix[j][i];
        matrix[i][j] = tmp;
    }

    /**
     * 49. Group Anagrams
     *
     * @param strs
     * @return
     */
    public List<List<String>> groupAnagrams(String[] strs) {
        if (strs == null || strs.length == 0) {
            return new ArrayList<>();
        }
        HashMap<String, List<String>> map = new HashMap<>();
        for (String tmp : strs) {
            char[] chars = tmp.toCharArray();
            Arrays.sort(chars);
            String key = String.valueOf(chars);
            if (!map.containsKey(key)) {
                map.put(key, new ArrayList<>());
            }
            map.get(key).add(tmp);
        }
        return new ArrayList<>(map.values());
    }

    /**
     * 50. Pow(x, n)
     *
     * @param x
     * @param n
     * @return
     */
    public double myPow(double x, int n) {
        double result = 1;
        int p = Math.abs(n);
        while (p != 0) {
            if ((p % 2) != 0) {
                result *= x;
            }
            x *= x;
            p /= 2;
        }
        if (result > Integer.MAX_VALUE || result < Integer.MIN_VALUE) {
            return 0;
        }
        return (n < 0) ? 1 / result : result;
    }

    /**
     * 51. N-Queens
     *
     * @param n
     * @return
     */
    public List<List<String>> solveNQueens(int n) {
        if (n < 0) {
            return new ArrayList<>();
        }
        char[][] nQueens = new char[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                nQueens[i][j] = '.';
            }
        }
        List<List<String>> ans = new ArrayList<>();
        this.solveNQueens(ans, nQueens, 0, n);
        return ans;

    }

    private void solveNQueens(List<List<String>> ans, char[][] nQueens, int row, int n) {
        if (row == n) {
            ans.add(this.construct(nQueens));
        }
        for (int col = 0; col < n; col++) {
            if (!this.checkExist(nQueens, col, row, n)) {
                nQueens[row][col] = 'Q';
                this.solveNQueens(ans, nQueens, row + 1, n);
                nQueens[row][col] = '.';
            }
        }
    }

    private boolean checkExist(char[][] nQueens, int col, int row, int n) {
        for (int i = 0; i < row; i++) {
            if (nQueens[i][col] == 'Q') {
                return true;
            }
        }
        for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--) {
            if (nQueens[i][j] == 'Q') {
                return true;
            }
        }
        for (int i = row - 1, j = col + 1; i >= 0 && j < n; i--, j++) {
            if (nQueens[i][j] == 'Q') {
                return true;
            }
        }
        return false;
    }

    private List<String> construct(char[][] nQueens) {
        List<String> ans = new ArrayList<>();
        for (char[] row : nQueens) {
            String tmp = String.valueOf(row);
            ans.add(tmp);
        }
        return ans;
    }

    /**
     * 52. N-Queens II
     *
     * @param n
     * @return
     */
    public int totalNQueens(int n) {
        if (n < 0) {
            return 0;
        }
        int[] dp = new int[n];
        return this.totalNQueens(dp, 0, n);
    }

    private int totalNQueens(int[] dp, int row, int n) {
        int result = 0;
        if (row == n) {
            result++;
            return result;
        }
        for (int col = 0; col < n; col++) {
            if (!this.checkExist(dp, row, col)) {
                dp[row] = col;
                result += this.totalNQueens(dp, row + 1, n);
                dp[row] = -1;
            }
        }
        return result;
    }

    private boolean checkExist(int[] dp, int row, int col) {
        for (int i = 0; i < row; i++) {
            if (dp[i] == col || Math.abs(dp[i] - i) == Math.abs(row - col)) {
                return true;
            }
        }
        return false;
    }

    /**
     * 53. Maximum Subarray
     *
     * @param nums
     * @return
     */
    public int maxSubArray(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int local = 0;
        int global = Integer.MIN_VALUE;
        for (int num : nums) {
            local = local > 0 ? local + num : num;
            global = Math.max(global, local);
        }
        return global;
    }

    /**
     * 54. Spiral Matrix
     *
     * @param matrix
     * @return
     */
    public List<Integer> spiralOrder(int[][] matrix) {
        if (matrix == null || matrix.length == 0) {
            return new ArrayList<>();
        }
        List<Integer> ans = new ArrayList<>();
        int left = 0;
        int right = matrix[0].length - 1;
        int top = 0;
        int bottom = matrix.length - 1;
        while (left <= right && top <= bottom) {
            for (int i = left; i <= right; i++) {
                ans.add(matrix[top][i]);
            }
            for (int i = top + 1; i <= bottom; i++) {
                ans.add(matrix[i][right]);
            }
            if (top != bottom) {
                for (int i = right - 1; i >= left; i--) {
                    ans.add(matrix[bottom][i]);
                }
            }
            if (left != right) {
                for (int i = bottom - 1; i > top; i--) {
                    ans.add(matrix[i][left]);
                }
            }
            left++;
            right--;
            top++;
            bottom--;
        }
        return ans;
    }

    /**
     * 55. Jump Game
     *
     * @param nums
     * @return
     */
    public boolean canJump(int[] nums) {
        if (nums == null || nums.length == 0) {
            return false;
        }
        int reach = nums[0];
        for (int i = 0; i < nums.length - 1 && i <= reach; i++) {
            reach = Math.max(reach, i + nums[i]);
        }
        return reach >= nums.length - 1;
    }

    public List<Interval> merge(List<Interval> intervals) {
        if (intervals == null || intervals.size() == 0) {
            return new ArrayList<>();
        }
        return null;
    }

    /**
     * 58. Length of Last Word
     *
     * @param s
     * @return
     */
    public int lengthOfLastWord(String s) {
        if (s == null || s.length() == 0) {
            return 0;
        }
        int lastIndex = s.lastIndexOf(" ");
        return s.length() - 1 - lastIndex;
    }

    /**
     * 59. Spiral Matrix II
     *
     * @param n
     * @return
     */
    public int[][] generateMatrix(int n) {
        int[][] matrix = new int[n][n];
        int total = 0;
        int left = 0;
        int right = n - 1;
        int top = 0;
        int bottom = n - 1;
        while (left <= right && top <= bottom) {
            for (int i = left; i <= right; i++) {
                matrix[top][i] = ++total;
            }
            for (int i = top + 1; i <= bottom; i++) {
                matrix[i][right] = ++total;
            }
            if (top != bottom) {
                for (int i = right - 1; i >= left; i--) {
                    matrix[bottom][i] = ++total;
                }
            }
            if (left != right) {
                for (int i = bottom - 1; i > left; i--) {
                    matrix[i][left] = ++total;
                }
            }
            left++;
            right--;
            top++;
            bottom--;
        }
        return matrix;
    }

    /**
     * 60. Permutation Sequence
     *
     * @param n
     * @param k
     * @return
     */
    public String getPermutation(int n, int k) {
        if (n <= 0) {
            return "";
        }
        int[] factory = new int[n];
        factory[0] = 1;
        int fac = 1;
        for (int i = 1; i < n; i++) {
            fac = i * fac;
            factory[i] = fac;
        }
        List<Integer> nums = new ArrayList<>();
        for (int i = 1; i <= n; i++) {
            nums.add(i);
        }
        k--;
        StringBuilder stringBuilder = new StringBuilder();
        for (int i = 0; i < n; i++) {
            int index = k / factory[n - 1 - i];
            stringBuilder.append(nums.get(index));
            nums.remove(index);
            k -= index * factory[n - 1 - i];
        }
        return stringBuilder.toString();
    }

    /**
     * 61. Rotate List
     *
     * @param head
     * @param k
     * @return
     */
    public ListNode rotateRight(ListNode head, int k) {
        if (head == null || k < 0) {
            return head;
        }
        ListNode fast = head;
        int count = 1;
        while (fast.next != null) {
            fast = fast.next;
            count++;
        }
        fast.next = head;
        ListNode slow = head;
        if ((k %= count) != 0) {
            for (int i = 0; i < count - k; i++) {
                slow = slow.next;
                fast = fast.next;
            }
        }
        fast.next = null;
        return slow;
    }

    /**
     * 62. Unique Paths
     *
     * @param m
     * @param n
     * @return
     */
    public int uniquePaths(int m, int n) {
        if (m < 0 || n < 0) {
            return 0;
        }
        int[] dp = new int[n];
        dp[0] = 1;
        for (int i = 0; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[j] += dp[j - 1];
            }
        }
        return dp[n - 1];
    }

    /**
     * 63. Unique Paths II
     *
     * @param obstacleGrid
     * @return
     */
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        if (obstacleGrid == null || obstacleGrid.length == 0) {
            return 0;
        }
        int column = obstacleGrid[0].length;
        int[] dp = new int[column];
        dp[0] = 1;
        for (int i = 0; i < obstacleGrid.length; i++) {
            for (int j = 0; j < column; j++) {
                if (obstacleGrid[i][j] == 1) {
                    dp[j] = 0;
                } else {
                    dp[j] += j > 0 ? dp[j - 1] : 0;
                }
            }
        }
        return dp[column - 1];
    }

    /**
     * 64. Minimum Path Sum
     *
     * @param grid
     * @return
     */
    public int minPathSum(int[][] grid) {
        if (grid == null || grid.length == 0) {
            return 0;
        }
        int row = grid.length;
        int column = grid[0].length;
        int[][] dp = new int[row][column];

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (i == 0 && j == 0) {
                    dp[0][0] = grid[i][j];
                } else if (i == 0) {
                    dp[i][j] = dp[i][j - 1] + grid[i][j];
                } else if (j == 0) {
                    dp[i][j] = dp[i - 1][j] + grid[i][j];
                } else {
                    dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
                }
            }
        }
        return dp[row - 1][column - 1];
    }

    /**
     * 66. Plus One
     *
     * @param digits
     * @return
     */
    public int[] plusOne(int[] digits) {
        int n = digits.length;
        for (int i = n - 1; i >= 0; i--) {
            if (digits[i] != 9) {
                digits[i]++;
                return digits;
            } else {
                digits[i] = 0;
            }
        }
        int[] ans = new int[n + 1];
        ans[0] = 1;
        return ans;
    }

    /**
     * 67. Add Binary
     *
     * @param a
     * @param b
     * @return
     */
    public String addBinary(String a, String b) {
        int m = a.length() - 1;
        int n = b.length() - 1;
        StringBuilder ans = new StringBuilder();
        int carry = 0;
        while (m >= 0 || n >= 0 || carry > 0) {
            int sum = (m >= 0 ? a.charAt(m--) - '0' : 0) + (n >= 0 ? b.charAt(n--) - '0' : 0) + carry;
            carry = sum / 2;
            ans.insert(0, sum % 2);
        }
        return ans.toString();
    }

    /**
     * 68. Text Justification
     *
     * @param words
     * @param maxWidth
     * @return
     */
    public List<String> fullJustify(String[] words, int maxWidth) {
        if (words == null || words.length == 0) {
            return new ArrayList<>();
        }
        List<String> ans = new ArrayList<>();
        int startIndex = 0;
        while (startIndex < words.length) {
            int line = 0;
            int endIndex = startIndex;
            while (endIndex < words.length && line + words[endIndex].length() <= maxWidth) {
                line += words[endIndex++].length() + 1;
            }
            boolean isLast = endIndex == words.length;
            int countOfWords = endIndex - startIndex;
            StringBuilder stringBuilder = new StringBuilder();
            if (countOfWords == 1) {
                stringBuilder.append(words[startIndex]);
            } else {
                int blankSpace = maxWidth - line + 1;
                int countOfSpace = isLast ? 1 : 1 + blankSpace / (countOfWords - 1);
                int extraSpace = isLast ? 0 : blankSpace % (countOfWords - 1);
                stringBuilder = this.construct(words, startIndex, endIndex, countOfSpace, extraSpace);
            }
            ans.add(this.correct(stringBuilder.toString(), maxWidth));
            startIndex = endIndex;
        }
        return ans;
    }

    private StringBuilder construct(String[] words, int startIndex, int endIndex, int spaces, int extra) {
        StringBuilder stringBuilder = new StringBuilder();
        for (int i = startIndex; i < endIndex; i++) {
            stringBuilder.append(words[i]);

            int tmp = 0;
            while (tmp < spaces) {
                stringBuilder.append(" ");
                tmp++;
            }
            if (extra > 0) {
                stringBuilder.append(" ");
                extra--;
            }
        }
        return stringBuilder;
    }

    private String correct(String stringBuilder, int maxWidth) {
        while (stringBuilder.length() < maxWidth) {
            stringBuilder = stringBuilder + " ";
        }
        while (stringBuilder.length() > maxWidth) {
            stringBuilder = stringBuilder.substring(0, stringBuilder.length() - 1);
        }
        return stringBuilder;
    }

    /**
     * 69. Sqrt(x)
     *
     * @param x
     * @return
     */
    public int mySqrt(int x) {
        double precision = 0.00001;
        double ans = x;
        while ((ans * ans - x) > precision) {
            ans = (ans + x / ans) / 2;
        }
        return (int) ans;
    }

    /**
     * 70. Climbing Stairs
     *
     * @param n
     * @return
     */
    public int climbStairs(int n) {
        if (n <= 2) {
            return n < 0 ? 0 : n;
        }
        int one = 1;
        int two = 2;
        int ans = 0;
        for (int i = 3; i <= n; i++) {
            ans = one + two;
            one = two;
            two = ans;
        }
        return ans;
    }

    /**
     * 71. Simplify Path
     *
     * @param path
     * @return
     */
    public String simplifyPath(String path) {
        if (path == null || path.length() == 0) {
            return "";
        }
        Deque<String> deque = new LinkedList<>();
        Set<String> skip = new HashSet<>(Arrays.asList("", "."));
        String[] paths = path.split("/");
        for (String tmp : paths) {
            if (!skip.contains(tmp)) {
                if ("..".equals(tmp)) {
                    if (!deque.isEmpty()) {
                        deque.pop();
                    }
                } else {
                    deque.push(tmp);
                }
            }
        }
        String ans = "";
        while (!deque.isEmpty()) {
            ans = "/" + deque.poll() + ans;
        }
        return ans.length() == 0 ? "/" : ans;
    }

    /**
     * 72. Edit Distance
     *
     * @param word1
     * @param word2
     * @return
     */
    public int minDistance(String word1, String word2) {
        if (word1 == null || word2 == null) {
            return 0;
        }
        int m = word1.length();
        int n = word2.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 0; i <= m; i++) {
            for (int j = 0; j <= n; j++) {
                if (i == 0 && j == 0) {
                    dp[i][j] = 0;
                } else if (i == 0) {
                    dp[i][j] = j;
                } else if (j == 0) {
                    dp[i][j] = i;
                } else {
                    if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                        dp[i][j] = dp[i - 1][j - 1];
                    } else {
                        dp[i][j] = 1 + Math.min(dp[i - 1][j - 1], Math.min(dp[i - 1][j], dp[i][j - 1]));
                    }
                }
            }
        }
        return dp[m][n];
    }

    /**
     * 73. Set Matrix Zeroes
     *
     * @param matrix
     */
    public void setZeroes(int[][] matrix) {
        if (matrix == null || matrix.length == 0) {
            return;
        }
        boolean fc = false;
        boolean fr = false;
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                if (matrix[i][j] == 0) {
                    matrix[i][0] = 0;
                    matrix[0][j] = 0;
                    if (i == 0) {
                        fc = true;
                    }
                    if (j == 0) {
                        fr = true;
                    }
                }
            }
        }
        for (int i = 1; i < matrix.length; i++) {
            for (int j = 1; j < matrix[i].length; j++) {
                if (matrix[i][0] == 0 || matrix[0][j] == 0) {
                    matrix[i][j] = 0;
                }
            }
        }
        if (fc) {
            for (int j = 0; j < matrix[0].length; j++) {
                matrix[0][j] = 0;
            }
        }
        if (fr) {
            for (int i = 0; i < matrix.length; i++) {
                matrix[i][0] = 0;
            }
        }
    }

    /**
     * 74. Search a 2D Matrix
     *
     * @param matrix
     * @param target
     * @return
     */
    public boolean searchMatrix(int[][] matrix, int target) {
        if (matrix == null || matrix.length == 0) {
            return false;
        }
        int row = matrix.length;
        int column = matrix[0].length;
        int j = 0;
        int i = row - 1;
        while (i >= 0 && j < column) {
            if (matrix[i][j] == target) {
                return true;
            } else if (matrix[i][j] < target) {
                j++;
            } else {
                i--;
            }
        }
        return false;
    }

    /**
     * 75. Sort Colors
     *
     * @param nums
     */
    public void sortColors(int[] nums) {
        if (nums == null || nums.length == 0) {
            return;
        }
        int one = 0;
        int second = nums.length - 1;
        for (int i = 0; i < nums.length; i++) {
            while (nums[i] == 2 && i < second) {
                this.swap(nums, i, second--);
            }
            while (nums[i] == 0 && i > one) {
                this.swap(nums, i, one++);
            }
        }
    }

    /**
     * 76. Minimum Window Substring
     *
     * @param s
     * @param t
     * @return
     */
    public String minWindow(String s, String t) {
        if (s == null || t == null) {
            return "";
        }
        HashMap<Character, Integer> map = new HashMap<>();
        for (int i = 0; i < s.length(); i++) {
            int count = map.getOrDefault(s.charAt(i), 0);
            map.put(s.charAt(i), ++count);
        }
        int begin = 0;
        int end = 0;
        int ans = 0;
        int count = t.length();
        while (end < s.length()) {

        }
        return "";
    }

    /**
     * 77. Combinations
     *
     * @param n
     * @param k
     * @return
     */
    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> ans = new ArrayList<>();
        this.combine(ans, new ArrayList<>(), 1, n, k);
        return ans;
    }

    private void combine(List<List<Integer>> ans, List<Integer> tmp, int start, int n, int k) {
        if (tmp.size() == k) {
            ans.add(new ArrayList<>(tmp));
        }
        for (int i = start; i <= n; i++) {
            tmp.add(i);
            this.combine(ans, tmp, i + 1, n, k);
            tmp.remove(tmp.size() - 1);
        }
    }

    /**
     * 78. Subsets
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> subsets(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new ArrayList<>();
        }
        List<List<Integer>> ans = new ArrayList<>();
        this.subsets(ans, new ArrayList<>(), 0, nums);
        return ans;
    }

    private void subsets(List<List<Integer>> ans, List<Integer> tmp, int index, int[] nums) {
        ans.add(new ArrayList<>(tmp));
        for (int i = index; i < nums.length; i++) {
            tmp.add(nums[i]);
            this.subsets(ans, tmp, i + 1, nums);
            tmp.remove(tmp.size() - 1);
        }
    }

    /**
     * 79. Word Search
     *
     * @param board
     * @param word
     * @return
     */
    public boolean exist(char[][] board, String word) {
        if (board == null || board.length == 0) {
            return false;
        }
        boolean[][] used = new boolean[board.length][board[0].length];
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length; j++) {
                if (board[i][j] == word.charAt(0) && this.check(used, i, j, board, 0, word)) {
                    return true;
                }
            }
        }
        return false;

    }

    private boolean check(boolean[][] used, int i, int j, char[][] board, int k, String word) {
        if (k == word.length()) {
            return true;
        }
        if (i < 0 || i >= board.length || j < 0 || j >= board[i].length || board[i][j] != word.charAt(k) || used[i][j]) {
            return false;
        }
        used[i][j] = true;
        if (this.check(used, i - 1, j, board, k + 1, word) ||
                this.check(used, i + 1, j, board, k + 1, word) ||
                this.check(used, i, j - 1, board, k + 1, word) ||
                this.check(used, i, j + 1, board, k + 1, word)) {
            return true;
        }
        used[i][j] = false;
        return false;
    }

    /**
     * 80. Remove Duplicates from Sorted Array II
     *
     * @param nums
     * @return
     */
    public int removeDuplicatesII(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int idx = 1;
        int count = 1;
        int value = nums[0];
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] == value) {
                count++;
                if (count > 2) {
                    continue;
                }
            } else {
                count = 1;
                value = nums[i];
            }
            nums[idx++] = nums[i];
        }
        return idx;
    }

    /**
     * 82. Remove Duplicates from Sorted List II
     *
     * @param head
     * @return
     */
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        if (head.val == head.next.val) {
            ListNode currNode = head.next.next;
            while (currNode != null && currNode.val == head.val) {
                currNode = currNode.next;
            }
            return this.deleteDuplicates(currNode);
        } else {
            head.next = this.deleteDuplicates(head.next);
            return head;
        }
    }

    /**
     * 83. Remove Duplicates from Sorted List
     *
     * @param head
     * @return
     */
    public ListNode deleteDuplicatesEasy(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        if (head.val == head.next.val) {
            return this.deleteDuplicatesEasy(head.next);
        } else {
            head.next = this.deleteDuplicatesEasy(head.next);
            return head;
        }
    }

    /**
     * 84. Largest Rectangle in Histogram
     *
     * @param heights
     * @return
     */
    public int largestRectangleArea(int[] heights) {
        if (heights == null || heights.length == 0) {
            return 0;
        }
        Deque<Integer> stack = new LinkedList<>();
        int ans = 0;
        for (int i = 0; i <= heights.length; i++) {
            int h = i == heights.length ? 0 : heights[i];
            if (stack.isEmpty() || heights[stack.peek()] < h) {
                stack.push(i);
            } else {
                int value = heights[stack.pop()];
                int size = stack.isEmpty() ? i : i - 1 - stack.peek();
                ans = Math.max(ans, value * size);
                i--;
            }
        }
        return ans;
    }

    /**
     * 85. Maximal Rectangle
     *
     * @param matrix
     * @return
     */
    public int maximalRectangle(char[][] matrix) {
        if (matrix == null || matrix.length == 0) {
            return 0;
        }
        int ans = 0;
        int m = matrix.length;
        int n = matrix[0].length;
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == '0') {
                    dp[i][j] = -1;
                    continue;
                }
                dp[i][j] = j;
                if (j > 0 && dp[i][j - 1] >= 0) {
                    dp[i][j] = dp[i][j - 1];
                }
                int width = j - dp[i][j] + 1;
                for (int k = i; k >= 0 && matrix[k][j] == '1'; k--) {
                    width = Math.min(width, j - dp[k][j] + 1);
                    ans = Math.max(ans, (i - k + 1) * width);
                }

            }
        }
        return ans;
    }

    /**
     * 86. Partition List
     *
     * @param head
     * @param x
     * @return
     */
    public ListNode partition(ListNode head, int x) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode root = new ListNode(0);
        ListNode small = root;
        ListNode dummy = new ListNode(0);
        ListNode big = dummy;
        while (head != null) {
            if (head.val < x) {
                small.next = head;
                small = small.next;
            } else {
                big.next = head;
                big = big.next;
            }
            head = head.next;
        }
        big.next = null;
        small.next = dummy.next;
        return root.next;
    }

    /**
     * 87. Scramble String
     *
     * @param s1
     * @param s2
     * @return
     */
    public boolean isScramble(String s1, String s2) {
        if (s1 == null || s2 == null) {
            return false;
        }
        if (s1.equals(s2)) {
            return true;
        }
        int m = s1.length();

        int[] hash = new int[26];
        for (int i = 0; i < m; i++) {
            hash[s1.charAt(i) - 'a']++;
            hash[s2.charAt(i) - 'a']--;
        }
        for (int i = 0; i < 26; i++) {

            if (hash[i] != 0) {
                return false;
            }
        }
        for (int i = 1; i < m; i++) {
            if (this.isScramble(s1.substring(0, i), s2.substring(0, i)) && this.isScramble(s1.substring(i), s2.substring(i))) {
                return true;
            }
            if (this.isScramble(s1.substring(0, i), s2.substring(m - i)) && this.isScramble(s1.substring(i), s2.substring(0, m - i))) {
                return true;
            }
        }
        return false;
    }

    /**
     * 88. Merge Sorted Array
     *
     * @param nums1
     * @param m
     * @param nums2
     * @param n
     */
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int k = m + n - 1;
        while (m >= 1 && n >= 1) {
            if (nums1[m - 1] > nums2[n - 1]) {
                nums1[k--] = nums1[m - 1];
                m--;
            } else {
                nums1[k--] = nums2[n - 1];
                n--;
            }
        }
        while (n >= 1) {
            nums1[k--] = nums2[n - 1];
            n--;
        }
    }

    /**
     * 90. Subsets II
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new ArrayList<>();
        }
        Arrays.sort(nums);
        List<List<Integer>> ans = new ArrayList<>();
        this.subsetsWithDup(ans, new ArrayList<>(), 0, nums);
        return ans;
    }

    private void subsetsWithDup(List<List<Integer>> ans, List<Integer> integers, int startIndex, int[] nums) {
        ans.add(new ArrayList<>(integers));
        for (int i = startIndex; i < nums.length; i++) {
            if (i > startIndex && nums[i] == nums[i - 1]) {
                continue;
            }
            integers.add(nums[i]);
            this.subsetsWithDup(ans, integers, i + 1, nums);
            integers.remove(integers.size() - 1);
        }
    }

    /**
     * 93. Restore IP Addresses
     *
     * @param s
     * @return
     */
    public List<String> restoreIpAddresses(String s) {
        if (s == null || s.length() == 0) {
            return new ArrayList<>();
        }
        List<String> ans = new ArrayList<>();
        for (int a = 0; a < 4; a++) {
            for (int b = a + 1; b <= 7; b++) {
                for (int c = b + 1; c <= 10; c++) {
                    for (int d = c + 1; d <= s.length(); d++) {
                        if (a + b + c + d == 12) {
                            continue;
                        }

                    }
                }
            }
        }
        return ans;
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
        List<Integer> ans = new ArrayList<>();
        TreeNode p = root;
        Stack<TreeNode> stack = new Stack<>();
        while (p != null || !stack.isEmpty()) {
            while (p != null) {
                stack.push(p);
                p = p.left;
            }
            p = stack.pop();
            ans.add(p.val);
            p = p.right;
        }
        return ans;
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
        return this.generateTrees(1, n);
    }

    private List<TreeNode> generateTrees(int left, int right) {

        List<TreeNode> ans = new ArrayList<>();
        if (left > right) {
            ans.add(null);
            return ans;
        }
        if (left == right) {
            TreeNode root = new TreeNode(left);
            ans.add(root);
            return ans;
        }
        for (int i = left; i <= right; i++) {
            List<TreeNode> leftList = this.generateTrees(left, i - 1);
            List<TreeNode> rightList = this.generateTrees(i + 1, right);
            for (TreeNode leftNode : leftList) {
                for (TreeNode rightNode : rightList) {
                    TreeNode root = new TreeNode(i);
                    root.left = leftNode;
                    root.right = rightNode;
                    ans.add(root);
                }
            }
        }
        return ans;
    }

    /**
     * 96. Unique Binary Search Trees
     *
     * @param n
     * @return
     */
    public int numTrees(int n) {
        if (n <= 1) {
            return n;
        }
        int[] dp = new int[n + 1];
        dp[0] = dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                dp[i] += dp[j - 1] * dp[i - j];
            }
        }
        return dp[n];

    }


    /**
     * 97. Interleaving String
     *
     * @param s1
     * @param s2
     * @param s3
     * @return
     */
    public boolean isInterleave(String s1, String s2, String s3) {
        int m = s1.length(), n = s2.length();
        if (m + n != s3.length()) {
            return false;
        }

        boolean[] dp = new boolean[n + 1];
        for (int i = 0; i <= m; i++) {
            for (int j = 0; j <= n; j++) {
                if (i == 0 && j == 0) {
                    dp[j] = true;
                } else if (i == 0) {
                    dp[j] = s2.charAt(j - 1) == s3.charAt(i + j - 1) && dp[j - 1];
                } else if (j == 0) {
                    dp[j] = s1.charAt(i - 1) == s3.charAt(i + j - 1) && dp[j];
                } else {
                    dp[j] = (dp[j] && s1.charAt(i - 1) == s3.charAt(i + j - 1)) || (dp[j - 1] && s2.charAt(j - 1) == s3.charAt(i + j - 1));
                }
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
        TreeNode p = root;
        TreeNode prev = null;
        while (!stack.isEmpty() || p != null) {
            while (p != null) {
                stack.push(p);
                p = p.left;
            }
            p = stack.pop();
            if (null == prev) {
                prev = p;
            } else {
                if (prev.val >= p.val) {
                    return false;
                }
                prev = p;
            }
            p = p.right;
        }
        return true;
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
        if (p.val == q.val) {
            return this.isSameTree(p.left, q.left) && this.isSameTree(p.right, q.right);
        }
        return false;
    }


}
