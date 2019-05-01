package org.dora.algorithm.leetcode;

import org.dora.algorithm.datastructe.ListNode;
import org.dora.algorithm.datastructe.TreeNode;

import java.util.*;

/**
 * @author dora
 * @date 2019-04-26
 */
public class FirstPage {


    public static void main(String[] args) {
        FirstPage firstPage = new FirstPage();
        firstPage.minWindow("bdab", "ab");
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
            return new int[]{};
        }
        int[] ans = new int[2];
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(target - nums[i])) {

                ans[0] = map.get(target - nums[i]);
                ans[1] = i;
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
        if (l1 == null || l2 == null) {
            return null;
        }
        int carry = 0;
        ListNode root = new ListNode(-1);
        ListNode dummy = root;
        while (l1 != null || l2 != null) {
            int value = (l1 == null ? 0 : l1.val) + (l2 == null ? 0 : l2.val) + carry;
            ListNode node = new ListNode(value % 10);
            carry /= 10;

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
        if (s == null || s.isEmpty()) {
            return 0;
        }
        HashMap<Character, Integer> map = new HashMap<>();
        int left = 0;
        int result = 0;
        for (int i = 0; i < s.length(); i++) {
            if (map.containsKey(s.charAt(i))) {
                left = Math.max(left, map.get(s.charAt(i)) + 1);
            }
            map.put(s.charAt(i), i);
            result = Math.max(result, i - left + 1);
        }
        return result;
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
        return 0;
    }

    /**
     * 5. Longest Palindromic Substring
     *
     * @param s
     * @return
     */
    public String longestPalindrome(String s) {
        if (s == null || s.isEmpty()) {
            return "";
        }
        int left = 0;
        int len = 0;
        int n = s.length();
        boolean[][] dp = new boolean[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                if (i - j <= 2) {
                    dp[j][i] = s.charAt(i) == s.charAt(j);
                } else {
                    dp[j][i] = s.charAt(i) == s.charAt(j) && dp[j + 1][i - 1];
                }
                if (dp[j][i] && i - j + 1 > len) {
                    left = j;
                    len = i - j + 1;
                }
            }
        }
        if (len > 0) {
            return s.substring(left, left + len);
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
        if (s == null || s.isEmpty() || numRows <= 1) {
            return "";
        }
        char[] chars = s.toCharArray();
        StringBuilder[] stringBuilders = new StringBuilder[numRows];
        for (int i = 0; i < numRows; i++) {
            stringBuilders[i] = new StringBuilder();
        }
        StringBuilder ans = new StringBuilder();
        int index = 0;
        while (index < chars.length) {
            for (int i = 0; i < numRows && index < chars.length; i++) {
                stringBuilders[i].append(chars[index++]);
            }
            for (int i = numRows - 2; i >= 1 && index < chars.length; i--) {
                stringBuilders[i].append(chars[index++]);
            }
        }
        for (int i = 1; i < numRows; i++) {
            stringBuilders[0].append(stringBuilders[i]);
        }
        return stringBuilders[0].toString();
    }

    /**
     * 7. Reverse Integer
     *
     * @param x
     * @return
     */
    public int reverse(int x) {
        int sum = 0;
        while (x != 0) {
            if (x > Integer.MAX_VALUE / 10 || x < Integer.MIN_VALUE / 10) {
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
        str = str.trim();
        if (str.isEmpty()) {
            return 0;
        }
        int sign = 1;

        int index = 0;

        while (index < str.length() & !Character.isDigit(str.charAt(index))) {
            if (str.charAt(index) == '+') {
                sign = 1;
                index++;
                break;
            } else if (str.charAt(index) == '-') {
                sign = -1;
                index++;
                break;
            } else {
                return 0;
            }
        }

        Long ans = 0L;
        while (index < str.length() && Character.isDigit(str.charAt(index))) {
            ans = ans * 10 + str.charAt(index) - '0';
            index++;
            if (ans > Integer.MAX_VALUE) {
                return sign == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
            }
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
        if (x != 0 && x % 10 == 0) {
            return false;
        }
        int sum = 0;
        while (x > sum) {
            sum = sum * 10 + x % 10;
            x /= 10;
        }
        return sum / 10 == x || sum == x;
    }

    /**
     * 10. Regular Expression Matching
     *
     * @param s
     * @param p
     * @return
     */
    public boolean isMatch(String s, String p) {
        if (s == null && p == null) {
            return false;
        } else if (s == null) {
            return true;
        }
        int m = s.length();
        int n = p.length();

        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[0][0] = true;
        for (int j = 1; j <= n; j++) {
            dp[0][j] = p.charAt(j - 1) == '*' && dp[0][j - 2];
        }

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (s.charAt(i - 1) == p.charAt(j - 1) || p.charAt(j - 1) == '.') {
                    dp[i][j] = dp[i - 1][j - 1];
                } else if (p.charAt(j - 1) == '*') {
                    if (s.charAt(i - 1) != p.charAt(j - 2) && p.charAt(j - 2) != '.') {
                        dp[i][j] = dp[i][j - 2];
                    } else {
                        dp[i][j] = dp[i][j - 1] || dp[i - 1][j] || dp[i][j - 2];
                    }
                }
            }
        }
        return dp[m][n];

    }

    /**
     * 11. Container With Most Water
     * two point move to middle
     * because the width is decrease,
     * we need chose the moving point according to height
     *
     * @param height
     * @return
     */
    public int maxArea(int[] height) {
        if (height == null || height.length == 0) {
            return 0;
        }
        int left = 0;
        int right = height.length - 1;
        int ans = 0;
        while (left < right) {
            ans = Math.max(ans, (right - left) * Math.min(height[left], height[right]));
            if (height[left] < height[right]) {
                left++;
            } else {
                right--;
            }
        }
        return ans;
    }

    /**
     * 12. Integer to Roman
     *
     * @param num
     * @return
     */
    public String intToRoman(int num) {
        return "";
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
        for (String str : strs) {
            while (!str.startsWith(prefix)) {
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
        Arrays.sort(nums);
        List<List<Integer>> ans = new ArrayList<>();
        for (int i = 0; i < nums.length - 2; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            int left = i + 1;
            int right = nums.length - 1;
            while (left < right) {
                int sum = nums[i] + nums[left] + nums[right];
                if (sum == 0) {
                    List<Integer> tmp = Arrays.asList(nums[i], nums[left], nums[right]);
                    ans.add(tmp);
                    while (left < right && nums[left] == nums[left + 1]) {
                        left++;
                    }
                    while (left < right && nums[right] == nums[right - 1]) {
                        right--;
                    }
                    left++;
                    right--;
                } else if (sum < 0) {
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
            return 0;
        }
        int result = nums[0] + nums[1] + nums[2];
        Arrays.sort(nums);
        for (int i = 0; i < nums.length - 2; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            int left = i + 1;
            int right = nums.length - 1;
            while (left < right) {
                int sum = nums[i] + nums[left] + nums[right];
                if (sum == target) {
                    return sum;
                } else if (sum < target) {
                    left++;
                } else {
                    right--;
                }
                if (Math.abs(result - target) > Math.abs(sum - target)) {
                    result = sum;
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
        if (digits == null || digits.isEmpty()) {
            return new ArrayList<>();
        }
        LinkedList<String> ans = new LinkedList<>();
        String[] map = new String[]{"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        ans.addLast("");
        for (int i = 0; i < digits.length(); i++) {
            int index = digits.charAt(i) - '0';
            String tmp = map[index];
            while (ans.peek().length() == i) {
                String pop = ans.pop();
                for (char c : tmp.toCharArray()) {
                    String value = pop + c;
                    ans.add(value);
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
        if (head == null) {
            return null;
        }
        ListNode root = new ListNode(0);
        root.next = head;
        ListNode fast = root;
        for (int i = 0; i < n; i++) {
            fast = fast.next;
        }
        ListNode slow = root;
        while (fast.next != null) {
            fast = fast.next;
            slow = slow.next;
        }
        ListNode tmp = slow.next;
        slow.next = tmp.next;
        tmp.next = null;
        return root.next;

    }

    /**
     * 20. Valid Parentheses
     *
     * @param s
     * @return
     */
    public boolean isValid(String s) {
        if (s == null || s.isEmpty()) {
            return false;
        }
        Stack<Character> stack = new Stack<>();
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') {
                stack.push(')');
            } else if (s.charAt(i) == '{') {
                stack.push('}');
            } else if (s.charAt(i) == '[') {
                stack.push(']');
            } else {
                if (!stack.isEmpty() && stack.peek() == s.charAt(i)) {
                    stack.pop();
                } else {
                    return false;
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
        } else if (l1 == null) {
            return l2;
        } else if (l2 == null) {
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
        this.generateParenthesis(ans, 0, 0, "", n);
        return ans;
    }

    private void generateParenthesis(List<String> ans, int open, int close, String s, int n) {
        if (n * 2 == s.length()) {
            ans.add(s);
        }
        if (open < n) {
            this.generateParenthesis(ans, open + 1, close, s + "(", n);
        }
        if (close < open) {
            this.generateParenthesis(ans, open, close + 1, s + ")", n);
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
        PriorityQueue<ListNode> queue = new PriorityQueue<>(lists.length, new Comparator<ListNode>() {
            @Override
            public int compare(ListNode o1, ListNode o2) {
                if (o1.val < o2.val) {
                    return -1;
                } else if (o1.val == o2.val) {
                    return 0;
                } else {
                    return 1;
                }
            }
        });
        for (ListNode listNode : lists) {
            if (listNode != null) {
                queue.add(listNode);
            }
        }
        ListNode root = new ListNode(0);
        ListNode dummy = root;
        while (!queue.isEmpty()) {
            ListNode node = queue.poll();
            dummy.next = node;
            dummy = dummy.next;
            if (node.next != null) {
                queue.add(node.next);
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

            ListNode tmp = fast.next;

            fast.next = slow;

            dummy.next = fast;

            slow.next = tmp;

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
        if (head == null || k <= 0) {
            return null;
        }
        ListNode fast = head;
        for (int i = 0; i < k; i++) {
            if (fast == null) {
                return head;
            }
            fast = fast.next;
        }
        ListNode newHead = this.reverseListNode(head, fast);
        head.next = this.reverseKGroup(head.next, k);
        return head;


    }

    private ListNode reverseListNode(ListNode start, ListNode end) {
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
     * 26. Remove Duplicates from Sorted Array
     *
     * @param nums
     * @return
     */
    public int removeDuplicates(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int index = 1;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] != nums[i - 1]) {
                nums[index++] = nums[i];
            }
        }
        return index;
    }

    /**
     * 27. Remove Element
     *
     * @param nums
     * @param val
     * @return
     */
    public int removeElement(int[] nums, int val) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int index = 0;
        for (int num : nums) {
            if (num != val) {
                nums[index++] = num;
            }
        }
        return index;
    }

    /**
     * 28. Implement strStr()
     *
     * @param haystack
     * @param needle
     * @return
     */
    public int strStr(String haystack, String needle) {
        for (int i = 0; i < haystack.length(); i++) {
            for (int j = 0; j < needle.length(); j++) {
                if (j == needle.length()) {
                    return i;
                }

            }
        }
        return 0;
    }

    /**
     * 29. Divide Two Integers
     *
     * @param dividend
     * @param divisor
     * @return
     */
    public int divide(int dividend, int divisor) {
        if (divisor == 0) {
            return Integer.MAX_VALUE;
        }
        if (dividend == Integer.MIN_VALUE && divisor == -1) {
            return Integer.MAX_VALUE;
        }
        if (divisor == 1) {
            return dividend;
        }
        int sign = (dividend < 0) ^ (divisor < 0) ? -1 : 1;

        int dvd = Math.abs(dividend);

        int dvs = Math.abs(divisor);

        int ans = 0;
        while (dvd >= dvs) {
            int tmp = 1;
            int multi = dvs;
            while (dvd >= (multi << 1)) {
                tmp <<= 1;
                multi <<= 1;
            }
            ans += tmp;
            dvd -= multi;
        }
        return ans;

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
        while (index > 0) {
            if (nums[index] > nums[index - 1]) {
                break;
            }
            index--;
        }
        if (index == 0) {
            this.reverseNums(nums, 0, nums.length - 1);
        } else {
            int j = nums.length - 1;
            while (j >= index) {
                if (nums[j] > nums[index - 1]) {
                    break;
                }
                j--;
            }
            ;
            this.swap(nums, index - 1, j);
            this.reverseNums(nums, index, nums.length - 1);

        }
    }

    private void reverseNums(int[] nums, int start, int end) {
        for (int i = start; i <= (start + end) / 2; i++) {
            this.swap(nums, i, start + end - i);
        }
    }

    private void swap(int[] nums, int start, int end) {
        int tmp = nums[start];
        nums[start] = nums[end];
        nums[end] = tmp;
    }

    /**
     * 32. Longest Valid Parentheses
     *
     * @param s
     * @return
     */
    public int longestValidParentheses(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        }
        int left = -1;
        int result = 0;
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') {
                stack.push(i);
            } else if (stack.isEmpty()) {
                left = i;
            } else {
                if (s.charAt(stack.peek()) == '(') {
                    stack.pop();
                }
                if (stack.isEmpty()) {
                    result = Math.max(result, i - left);
                } else {
                    result = Math.max(result, i - stack.peek());
                }

            }

        }
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
        while (left <= right) {

            int mid = left + (right - left) / 2;


            if (nums[mid] == target) {
                return mid;
            } else if (nums[left] <= nums[mid]) {
                if (target < nums[mid] && nums[left] <= target) {
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
        return -1;
    }

    /**
     * 34. Find First and Last Position of Element in Sorted Array
     * tricker : 两次二分搜索
     *
     * @param nums
     * @param target
     * @return
     */
    public int[] searchRange(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return new int[]{-1, -1};
        }
        int[] ans = new int[]{-1, -1};

        // 方案一、两次二分搜索
        // 第一次从右往左
        // 第二次从左往右
//        int left = 0;
//        int right = nums.length - 1;
//        while (left < right) {
//            int mid = left + (right - left) / 2;
//            if (nums[mid] < target) {
//                left = mid + 1;
//            } else {
//                right = mid;
//            }
//        }
//        if (nums[left] != target) {
//            return ans;
//        }
//        ans[0] = left;
//        right = nums.length - 1;
//        while (left < right) {
//            int mid = left + (right - left) / 2 + 1;
//            if (nums[mid] > target) {
//                right = mid - 1;
//            } else {
//                left = mid;
//            }
//        }
//        ans[1] = left;

        // 方案二
        int left = 0;
        int right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                int startIndex = mid;
                while (startIndex > 0 && nums[startIndex] == nums[startIndex - 1]) {
                    startIndex--;
                }

                int endIndex = mid;
                while (endIndex < nums.length - 1 && nums[endIndex] == nums[endIndex + 1]) {
                    endIndex++;
                }
                ans[0] = startIndex;

                ans[1] = endIndex;

                return ans;
            } else if (nums[mid] < target) {
                left++;
            } else {
                right--;
            }
        }
        return ans;
    }

    /**
     * 35. Search Insert Position
     *
     * @param nums
     * @param target
     * @return
     */
    public int searchInsert(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int left = 0;
        int right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return left;
    }

    /**
     * 39. Combination Sum
     *
     * @param candidates
     * @param target
     * @return
     */
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        if (candidates == null || candidates.length == 0) {
            return new ArrayList<>();
        }
        Arrays.sort(candidates);
        List<List<Integer>> ans = new ArrayList<>();
        this.combinationSum(ans, new ArrayList<>(), 0, candidates, target);
        return ans;
    }

    private void combinationSum(List<List<Integer>> ans, List<Integer> tmp, int start, int[] candidates, int target) {
        if (target == 0) {
            ans.add(new ArrayList<>(tmp));
            return;
        }
        for (int i = start; i < candidates.length && candidates[i] <= target; i++) {
            if (i > start && candidates[i] == candidates[i - 1]) {
                continue;
            }
            tmp.add(candidates[i]);
            this.combinationSum(ans, tmp, i, candidates, target - candidates[i]);
            tmp.remove(tmp.size() - 1);
        }
    }

    /**
     * 40. Combination Sum II
     *
     * @param candidates
     * @param target
     * @return
     */
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        if (candidates == null || candidates.length == 0) {
            return new ArrayList<>();
        }
        Arrays.sort(candidates);
        List<List<Integer>> ans = new ArrayList<>();
        this.combinationSum2(ans, new ArrayList<>(), 0, candidates, target);
        return ans;

    }

    private void combinationSum2(List<List<Integer>> ans, List<Integer> tmp, int start, int[] candidates, int target) {
        if (target == 0) {
            ans.add(new ArrayList<>(tmp));
            return;
        }
        for (int i = start; i < candidates.length && candidates[i] <= target; i++) {
            if (i > start && candidates[i] == candidates[i - 1]) {
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
        if (nums == null || nums.length == 0) {
            return 0;
        }
        for (int i = 0; i < nums.length; i++) {
            while (nums[i] > 0 && nums[i] <= nums.length && nums[i] != nums[nums[i] - 1]) {
                this.swap(nums, i, nums[i] - 1);
            }
        }
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != i + 1) {
                return i;
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
        if (height == null || height.length == 0) {
            return 0;
        }
        int left = 0;
        int right = height.length - 1;
        int result = 0;

        int minLeft = 0;
        int minRight = 0;

        while (left < right) {
            if (height[left] <= height[right]) {
                if (height[left] >= minLeft) {
                    minLeft = height[left];
                } else {
                    result += minLeft - height[left];

                }
                left++;
            } else {
                if (height[right] >= minRight) {
                    minRight = height[right];
                } else {
                    result += minRight - height[right];
                }
                right--;
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
        int m = num1.length();
        int n = num2.length();

        int[] nums = new int[m + n];
        int carry = 0;
        for (int i = m - 1; i >= 0; i--) {
            for (int j = n - 1; j >= 0; j--) {
                int value = (num1.charAt(i) - '0') * (num2.charAt(j) - '0') + nums[i + 1 + j];

                nums[i + 1 + j] = value % 10;
                nums[i + j] += value / 10;
            }
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != 0 || (sb.length() != 0)) {
                sb.append(nums[i]);
            }
        }
        return sb.length() == 0 ? "0" : sb.toString();
    }

    /**
     * 44. Wildcard Matching
     * todo 不懂
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

        for (int j = 1; j <= n; j++) {
            dp[0][j] = p.charAt(j - 1) == '*';
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (s.charAt(i - 1) == p.charAt(j - 1) || p.charAt(j - 1) == '?') {
                    dp[i][j] = dp[i - 1][j - 1];
                } else if (p.charAt(j - 1) == '*') {
                    dp[i][j] = dp[i - 1][j] || dp[i][j - 1] || dp[i][j - 2];
                }
            }
        }

        return dp[m][n];
    }

    /**
     * 45. Jump Game II
     * trick : 贪心算法
     *
     * @param nums
     * @return
     */
    public int jump(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int step = 0;
        int furthest = 0;
        int current = 0;
        for (int i = 0; i < nums.length - 1; i++) {
            furthest = Math.max(furthest, i + nums[i]);
            if (i == current) {
                step++;
                current = furthest;
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
        this.permute(ans, new ArrayList<>(), nums, used);
        return ans;


    }

    private void permute(List<List<Integer>> ans, List<Integer> tmp, int[] nums, boolean[] used) {
        if (tmp.size() == nums.length) {
            ans.add(new ArrayList<>(tmp));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (used[i]) {
                continue;
            }
            tmp.add(nums[i]);
            used[i] = true;
            this.permute(ans, tmp, nums, used);
            tmp.remove(tmp.size() - 1);
            used[i] = false;

        }

    }

    /**
     * 47. Permutations II
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> permuteUnique(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new ArrayList<>();
        }
        Arrays.sort(nums);
        List<List<Integer>> ans = new ArrayList<>();
        boolean[] used = new boolean[nums.length];
        this.permuteUnique(ans, new ArrayList<Integer>(), nums, used);
        return ans;
    }

    private void permuteUnique(List<List<Integer>> ans, List<Integer> tmp, int[] nums, boolean[] used) {
        if (tmp.size() == nums.length) {
            ans.add(new ArrayList<>(tmp));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (used[i]) {
                continue;
            }
            if (i > 0 && nums[i] == nums[i - 1] && !used[i - 1]) {
                continue;
            }
            tmp.add(nums[i]);
            used[i] = true;
            this.permuteUnique(ans, tmp, nums, used);
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
            for (int j = 0; j < i; j++) {
                this.swapMatrix(matrix, i, j);
            }
        }
        for (int i = 0; i < matrix.length; i++) {

            int[] row = matrix[i];

            int start = 0;

            int end = row.length - 1;

            for (int j = start; j <= (start + end) / 2; j++) {
                this.swap(row, j, start + end - j);
            }
        }

    }

    private void swapMatrix(int[][] matrix, int i, int j) {
        int tmp = matrix[i][j];
        matrix[i][j] = matrix[j][i];
        matrix[j][i] = tmp;
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

        List<List<String>> ans = new ArrayList<>();

        HashMap<String, List<String>> map = new HashMap<>();

        for (String str : strs) {
            char[] chars = str.toCharArray();

            Arrays.sort(chars);
            String key = String.valueOf(chars);
            if (!map.containsKey(key)) {
                map.put(key, new ArrayList<>());
            }
            map.get(key).add(str);
        }
        ans.addAll(map.values());
        return ans;

    }

    /**
     * 50. Pow(x, n)
     *
     * @param x
     * @param n
     * @return
     */
    public double myPow(double x, int n) {
        if (n == 0) {
            return 1;
        }
        if (n < 0) {
            x = 1 / x;
            n = -n;
        }
        if (x > Integer.MAX_VALUE || x < Integer.MIN_VALUE) {
            return 0;
        }
        return (n & 1) == 0 ? this.myPow(x * x, n / 2) : x * this.myPow(x * x, n / 2);
    }

    /**
     * 51. N-Queens
     *
     * @param n
     * @return
     */
    public List<List<String>> solveNQueens(int n) {
        if (n <= 0) {
            return new ArrayList<>();
        }
        char[][] nQueens = new char[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                nQueens[i][j] = '.';
            }
        }

        List<List<String>> ans = new ArrayList<>();

        for (int i = 0; i < n; i++) {

            this.getQueens(nQueens, ans, i, n);
        }
        return ans;
    }

    private void getQueens(char[][] nQueens, List<List<String>> ans, int row, int n) {
        if (row == n) {
            List<String> tmp = new ArrayList<>();
            for (char[] nQueen : nQueens) {
                tmp.add(String.valueOf(nQueen));
            }
            ans.add(tmp);
            return;
        }
        for (int j = 0; j < n; j++) {
            if (this.verifyNQueens(nQueens, row, j, n)) {
                nQueens[row][j] = 'Q';
                this.getQueens(nQueens, ans, row + 1, n);
                nQueens[row][j] = ',';
            }
        }

    }

    private boolean verifyNQueens(char[][] nQueens, int row, int column, int n) {
        for (int i = row - 1; i >= 0; i--) {
            if (nQueens[i][column] == 'Q') {
                return false;
            }
        }
        for (int i = row - 1, j = column - 1; i >= 0 && j >= 0; i--, j--) {
            if (nQueens[i][j] == 'Q') {
                return false;
            }

        }
        for (int i = row - 1, j = column + 1; i >= 0 && j < n; i--, j++) {
            if (nQueens[i][j] == 'Q') {
                return false;
            }
        }
        return true;
    }


    /**
     * 52. N-Queens II
     * trick : 每行所在下标 用dp 存储起来
     *
     * @param n
     * @return
     */
    public int totalNQueens(int n) {
        if (n <= 0) {
            return 0;
        }
        int[] dp = new int[n];
        int count = this.getTotal(dp, 0);
        return count;
    }

    private int getTotal(int[] dp, int row) {
        int count = 0;
        if (row == dp.length) {
            count++;
            return 1;
        }

        for (int j = 0; j < dp.length; j++) {
            if (this.verify(dp, row, j)) {
                dp[row] = j;
                count += this.getTotal(dp, row + 1);

                dp[row] = -1;
            }
        }

        return count;
    }

    private boolean verify(int[] dp, int row, int column) {
        for (int i = row - 1; i >= 0; i--) {
            if (dp[i] == column || Math.abs(i - row) == Math.abs(column - dp[i])) {
                return false;
            }
        }
        return true;
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
        for (int i = 0; i < nums.length; i++) {
            local = local > 0 ? local + nums[i] : nums[i];
            global = Math.max(local, global);
        }
        return global;
    }


    /**
     * @param matrix
     * @return
     */
    public List<Integer> spiralOrder(int[][] matrix) {
        if (matrix == null || matrix.length == 0) {
            return new ArrayList<>();
        }
        List<Integer> ans = new ArrayList<>();
        int row = matrix.length;
        int column = matrix[0].length;
        int left = 0;
        int right = column - 1;
        int top = 0;
        int bottom = row - 1;
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
        int reach = Integer.MAX_VALUE;
        for (int i = 0; i < nums.length - 1 && i <= reach; i++) {

        }
        return reach >= nums.length - 1;
    }

    /**
     * 57. Insert Interval
     *
     * @param intervals
     * @param newInterval
     * @return
     */
    public int[][] insert(int[][] intervals, int[] newInterval) {
        if (intervals == null || intervals == null) {
            return new int[][]{};
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
        s = s.trim();
        if (s.isEmpty()) {
            return 0;
        }
        int lastIndex = s.length() - 1;
        int count = 0;
        while (lastIndex >= 0 && s.charAt(lastIndex) != ' ') {
            count++;
            lastIndex--;
        }
        return count;
    }

    /**
     * 59. Spiral Matrix II
     *
     * @param n
     * @return
     */
    public int[][] generateMatrix(int n) {
        if (n <= 0) {
            return new int[][]{};
        }
        int[][] matrix = new int[n][n];
        int left = 0;
        int right = n - 1;
        int top = 0;
        int bottom = n - 1;
        int total = 0;
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
                for (int i = bottom - 1; i > top; i--) {
                    matrix[i][left] = ++total;
                }
            }
            top++;
            left++;
            right--;
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

        int[] array = new int[n + 1];

        int base = 1;

        array[0] = 1;

        for (int i = 1; i <= n; i++) {


            base *= i;

            array[i] = base;

        }
        List<Integer> nums = new ArrayList<>();
        for (int i = 1; i <= n; i++) {
            nums.add(i);
        }
        List<Integer> ans = new ArrayList<>();
        k--;
        for (int i = 0; i < n; i++) {
            int index = k / array[n - 1 - i];
            ans.add(nums.get(index));
            nums.remove(index);
            k -= index * array[n - 1 - i];
        }
        StringBuilder stringBuilder = new StringBuilder();
        for (Integer number : ans) {
            stringBuilder.append(number);
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
        if (head == null || head.next == null || k == 0) {
            return head;
        }

        int count = 1;

        ListNode fast = head;
        while (fast.next != null) {
            fast = fast.next;
            count++;
        }
        fast.next = head;
        ListNode slow = head;
        if ((k %= count) != 0) {
            for (int i = 0; i < count - k; i++) {
                fast = fast.next;
                slow = slow.next;
            }
        }
        ListNode tmp = slow.next;
        slow.next = null;

        return tmp;

    }


    /**
     * 62. Unique Paths
     *
     * @param m
     * @param n
     * @return
     */
    public int uniquePaths(int m, int n) {
        int[] dp = new int[n];
        dp[0] = 1;
        for (int i = 0; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[j] = dp[j - 1] + dp[j];
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
        int row = obstacleGrid.length;
        int column = obstacleGrid[0].length;
        int[] dp = new int[column];
        dp[0] = 1;
        for (int i = 0; i < row; i++) {

            for (int j = 0; j < column; j++) {
                if (obstacleGrid[i][j] == 1) {
                    dp[j] = 0;
                } else if (j > 0) {
                    dp[j] = dp[j - 1] + dp[j];
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
                    dp[i][j] = grid[0][0];
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
     * 65. Valid Number
     *
     * @param s
     * @return
     */
    public boolean isNumber(String s) {
        if (s == null || s.length() == 0) {
            return false;
        }
        s = s.trim();

        int index = 0;
        boolean hasSign = false;
        while (index < s.length() && !Character.isDigit(s.charAt(index))) {
            if (s.charAt(index) == '+' || s.charAt(index) == '-') {
                if (hasSign) {
                    return false;
                }
                hasSign = true;
                index++;
            } else {
                return false;
            }
        }
        while (index < s.length() && Character.isDigit(s.charAt(index))) {
            index++;
        }
        if (index == s.length()) {
            return true;
        }
        return false;

    }

    /**
     * 66. Plus One
     *
     * @param digits
     * @return
     */
    public int[] plusOne(int[] digits) {
        if (digits == null || digits.length == 0) {
            return new int[]{};
        }
        for (int i = digits.length - 1; i >= 0; i--) {
            if (digits[i] != 9) {
                digits[i]++;
                return digits;
            } else {
                digits[i] = 0;
            }
        }
        int[] ans = new int[digits.length + 1];
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
        String ans = "";
        int m = a.length() - 1;
        int n = b.length() - 1;
        int carry = 0;
        while (m >= 0 || n >= 0 || carry > 0) {
            int value = (m >= 0 ? a.charAt(m--) - '0' : 0) + (n >= 0 ? b.charAt(n--) - '0' : 0) + carry;

            ans = ((char) (value % 2 + '0')) + ans;

            carry = value / 2;
        }
        return ans;
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

                line += words[endIndex].length() + 1;

                endIndex++;
            }

            StringBuilder sb = new StringBuilder();


            boolean lastRow = endIndex == words.length;

            int lineOfRow = maxWidth - line + 1;

            int numOfWord = endIndex - startIndex;

            if (numOfWord == 1) {
                sb.append(words[startIndex]);
            } else {
                int blankWord = lastRow ? 1 : 1 + (maxWidth - line + 1) / (numOfWord - 1);

                int extraWord = lastRow ? 0 : (maxWidth - line + 1) % (numOfWord - 1);

                this.constructRow(sb, blankWord, extraWord, startIndex, endIndex, words);
            }

            startIndex = endIndex;

            ans.add(this.adjustString(sb, maxWidth));

        }
        return ans;
    }

    private String adjustString(StringBuilder sb, int maxWidth) {
        String value = sb.toString();
        while (value.length() > maxWidth) {
            value = value.substring(0, value.length() - 1);
        }
        while (value.length() < maxWidth) {
            value = value + " ";
        }
        return value;
    }

    private void constructRow(StringBuilder sb, int countOfBlank, int extraOfBlank,
                              int startIndex, int endIndex, String[] words) {
        for (int i = startIndex; i < endIndex; i++) {
            sb.append(words[i]);
            int tmp = countOfBlank;
            while (tmp-- > 0) {
                sb.append(" ");
            }
            if (extraOfBlank-- > 0) {
                sb.append(" ");
            }
        }

    }


    /**
     * 69. Sqrt(x)
     *
     * @param x
     * @return
     */
    public int mySqrt(int x) {
        double precision = 0.0001;
        double ans = x;
        while (ans * ans - x > precision) {
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
        if (n == 0) {
            return 0;
        }
        if (n == 1) {
            return 1;
        }
        if (n == 2) {
            return 2;
        }
        int sum1 = 1;
        int sum2 = 2;
        int sum = 0;
        for (int i = 3; i <= n; i++) {
            sum = sum1 + sum2;
            sum1 = sum2;
            sum2 = sum;
        }
        return sum;
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
        Set<String> skip = new HashSet<>(Arrays.asList(".", "", ".."));
        Deque<String> ans = new LinkedList<>();
        String[] paths = path.split("/");
        for (int i = 0; i < paths.length; i++) {
            if (!skip.contains(paths[i])) {
                ans.add(paths[i]);
            } else if (!ans.isEmpty() && paths[i].equals("..")) {
                ans.pollLast();
            }
        }
        if (ans.isEmpty()) {
            return "/";
        }
        String sb = "";
        for (String str : ans) {
            sb = sb + "/" + str;
        }
        return sb;
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
                } else if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = 1 + Math.min(Math.min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]);
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
        boolean fr = false;
        boolean fc = false;
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
        int i = row - 1;
        int j = 0;
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
     * trick: 双指针
     *
     * @param nums
     */
    public void sortColors(int[] nums) {
        if (nums == null || nums.length == 0) {
            return;
        }
        int red = 0;
        int blue = nums.length - 1;
        for (int i = 0; i < nums.length; i++) {
            while (nums[i] == 2 && blue > i) {
                this.swap(nums, blue--, i);
            }
            while (nums[i] == 0 && i > red) {
                this.swap(nums, red++, i);
            }
        }
    }


    /**
     * 76. Minimum Window Substring
     * todo 未解
     *
     * @param s
     * @param t
     * @return
     */
    public String minWindow(String s, String t) {
        if (s == null || t == null) {
            return "";
        }
        int end = 0;

        int begin = 0;

        int result = Integer.MAX_VALUE;

        int head = 0;

        int count = t.length();

        int[] hash = new int[256];

        for (int i = 0; i < t.length(); i++) {
            hash[t.charAt(i) - '0']++;
        }

        while (end < s.length()) {

            while (end < s.length() && hash[s.charAt(end) - '0']-- > 0) {

                end++;
                count--;

            }
            while (count == 0) {
                if (end - begin < result) {
                    head = begin;
                    result = end - begin;
                }
                int index = s.charAt(begin) - '0';

                int value = hash[s.charAt(begin) - '0'];

                if (hash[s.charAt(begin++) - '0']++ == 0) {

                    count++;
                }
            }
            end++;
        }
        if (result < Integer.MAX_VALUE) {
            return s.substring(head, head + result);
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
        if (n <= 0) {
            return new ArrayList<>();
        }
        List<List<Integer>> ans = new ArrayList<>();
        this.combine(ans, new ArrayList<Integer>(), 1, n, k);
        return ans;
    }

    private void combine(List<List<Integer>> ans, List<Integer> tmp, int start, int n, int k) {
        if (tmp.size() == k) {
            ans.add(new ArrayList<>(tmp));
            return;
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

    private <E> void subsets(List<List<Integer>> ans, List<Integer> tmp, int start, int[] nums) {
        ans.add(new ArrayList<>(tmp));
        for (int i = start; i < nums.length; i++) {
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

        int row = board.length;

        int column = board[0].length;

        boolean[][] used = new boolean[row][column];
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (board[i][j] == word.charAt(0) && this.verify(board, i, j, used, 0, word)) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean verify(char[][] board, int i, int j, boolean[][] used, int k, String word) {
        if (k == word.length()) {
            return true;
        }
        if (i < 0 || i >= board.length || j < 0 || j >= board[i].length || used[i][j] || board[i][j] != word.charAt(k)) {
            return false;
        }
        used[i][j] = true;
        if (this.verify(board, i - 1, j, used, k + 1, word) ||
                this.verify(board, i + 1, j, used, k + 1, word) ||
                this.verify(board, i, j - 1, used, k + 1, word) ||
                this.verify(board, i, j + 1, used, k + 1, word)) {
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
        int count = 1;
        int index = 1;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] == nums[i - 1]) {
                count++;
                if (count >= 3) {
                    continue;
                }
            } else {
                count = 1;
            }
            nums[index++] = nums[i];
        }
        return index;
    }


    /**
     * 81. Search in Rotated Sorted Array II
     *
     * @param nums
     * @param target
     * @return
     */
    public boolean searchII(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return false;
        }
        int left = 0;
        int right = nums.length - 1;
        // 二分搜索关键
        // 每次省略一半部分
        // 故关键点在于 中值 以及中值比较部分
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                return true;
            } else if (nums[left] == nums[mid]) {
                left++;
            } else if (nums[left] < nums[mid]) {
                if (nums[left] <= target && target < nums[mid]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            } else {
                if (nums[mid] < target && target <= nums[right]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }
        return false;
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
            ListNode node = head.next.next;
            while (node != null && node.val == head.val) {
                node = node.next;
            }
            return this.deleteDuplicates(node);
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
    public ListNode deleteDuplicatesII(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        if (head.val == head.next.val) {
            return this.deleteDuplicates(head.next);

        } else {
            head.next = this.deleteDuplicates(head.next);
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
        Stack<Integer> stack = new Stack<>();
        int left = 0;
        int result = 0;
        for (int i = 0; i <= heights.length; i++) {
            int h = i == heights.length ? 0 : heights[i];
            if (stack.isEmpty() || heights[stack.peek()] < h) {
                stack.push(i);
            } else {
                int index = stack.pop();
                int size = stack.isEmpty() ? i : i - stack.peek() - 1;
                result = Math.max(result, heights[index] * size);
                i--;
            }
        }
        return result;
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
        int row = matrix.length;
        int column = matrix[0].length;

        // 使用三个一维数组来存储每列数组代表的值
        int[] left = new int[column];

        int[] right = new int[column];


        int[] height = new int[column];


        // 由于右边界从右往左比较 故需初始化
        for (int j = 0; j < column; j++) {
            right[j] = column;
        }


        int result = 0;

        // area = height[j] * (right[j] - left[j] if matrix[i][j] == '1'
        for (int i = 0; i < row; i++) {

            int maxLeft = 0;

            int minRight = column;

            for (int j = 0; j < column; j++) {
                if (matrix[i][j] == '0') {
                    height[j] = 0;
                } else {
                    height[j]++;
                }
            }
            for (int j = 0; j < column; j++) {
                if (matrix[i][j] == '1') {
                    left[j] = Math.max(left[j], maxLeft);
                } else {
                    left[j] = 0;
                    maxLeft = j + 1;
                }
            }
            for (int j = column - 1; j >= 0; j--) {
                if (matrix[i][j] == '1') {
                    right[j] = Math.min(right[j], minRight);

                } else {
                    right[j] = column;
                    minRight = j;
                }
            }
            for (int j = 0; j < column; j++) {
                result = Math.max(result, height[j] * (right[j] - left[j]));
            }
        }
        return result;
    }


    /**
     * 86. Partition List
     * todo 需慎重考虑
     *
     * @param head
     * @param x
     * @return
     */
    public ListNode partition(ListNode head, int x) {
        if (head == null) {
            return null;
        }
        ListNode root = new ListNode(0);

        ListNode big = root;
        ListNode node = big;
        ListNode small = root;
        while (head != null) {
            if (head.val <= x) {

                small.next = head;

                small = small.next;
            } else {

                big.next = head;

                big = big.next;
            }
            head = head.next;
        }

        small.next = node.next;

        return root.next;


    }


    /**
     * 87. Scramble String
     * todo 待优化
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
        int n = s2.length();
        if (m != n) {
            return false;
        }

        int[] hash = new int[256];
        for (int i = 0; i < m; i++) {
            hash[s1.charAt(i) - '0']--;
            hash[s2.charAt(i) - '0']++;
        }
        for (int i = 0; i < 256; i++) {
            if (hash[i] != 0) {
                return false;
            }
        }
        for (int i = 1; i < m; i++) {
            if (this.isScramble(s1.substring(0, i), s2.substring(0, i)) && this.isScramble(s1.substring(i), s2.substring(i))) {
                return true;
            }
            if (this.isScramble(s1.substring(i), s2.substring(0, m - i)) && this.isScramble(s1.substring(0, i), s2.substring(m - i))) {
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
        if (nums1 == null || nums2 == null) {
            return;
        }
        int s = m + n - 1;
        m = m - 1;
        n = n - 1;
        while (m >= 0 && n >= 0) {
            if (nums1[m] >= nums2[n]) {
                nums1[s--] = nums1[m--];
            } else {
                nums1[s--] = nums2[n--];
            }
        }
        while (n >= 0) {
            nums1[s--] = nums2[n--];
        }
    }


    /**
     * gray code
     * // todo 格雷码
     *
     * @param n
     * @return
     */
    public List<Integer> grayCode(int n) {
        return null;
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
        this.subsetsWithDup(ans, new ArrayList<Integer>(), 0, nums);
        return ans;
    }

    private void subsetsWithDup(List<List<Integer>> ans, List<Integer> tmp, int start, int[] nums) {
        ans.add(new ArrayList<>(tmp));
        for (int i = start; i < nums.length; i++) {
            if (i > start && nums[i] == nums[i - 1]) {
                continue;
            }
            tmp.add(nums[i]);
            this.subsetsWithDup(ans, tmp, i + 1, nums);
            tmp.remove(tmp.size() - 1);
        }
    }


    /**
     * 92. Reverse Linked List II
     *
     * @param head
     * @param m
     * @param n
     * @return
     */
    public ListNode reverseBetween(ListNode head, int m, int n) {
        if (head == null) {
            return null;
        }
        ListNode root = new ListNode(0);
        root.next = head;
        ListNode fast = root;
        ListNode slow = root;
        for (int i = 0; i < m - 1; i++) {
            slow = slow.next;
        }
        for (int j = 0; j < n; j++) {
            fast = fast.next;
        }

        ListNode prev = fast.next;

        ListNode current = slow.next;
        for (int i = 0; i <= n - m; i++) {

            ListNode tmp = current.next;
            current.next = prev;
            prev = current;
            current = tmp;

        }
        slow.next = prev;

        return root.next;
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
        return null;
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
        Stack<TreeNode> stack = new Stack<>();
        TreeNode node = root;
        while (!stack.isEmpty() || node != null) {
            while (node != null) {
                stack.push(node);
                node = node.left;
            }
            node = stack.pop();
            ans.add(node.val);
            node = node.right;
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

    private List<TreeNode> generateTrees(int start, int end) {
        List<TreeNode> list = new ArrayList<>();
        if (start > end) {
            list.add(null);
            return list;

        }
        if (start == end) {
            TreeNode node = new TreeNode(start);
            list.add(node);
            return list;
        }
        for (int i = start; i <= end; i++) {
            List<TreeNode> lefts = this.generateTrees(start, i - 1);
            List<TreeNode> rights = this.generateTrees(i + 1, end);
            for (TreeNode left : lefts) {
                for (TreeNode right : rights) {
                    TreeNode root = new TreeNode(i);
                    root.left = left;
                    root.right = right;
                    list.add(root);
                }
            }
        }
        return list;
    }

    /**
     * 96. Unique Binary Search Trees
     * trick : 动态规划
     * f(n) = r(1) + r(2) + r(3) + ... + r(n)
     * r(i) = f(i-1) * f(n-i);
     * so f(n) = r(0) * r(n-1) + r(1) * (n-1) + ..r(n-1) * r(0)
     *
     * @param n
     * @return
     */
    public int numTrees(int n) {
        if (n <= 1) {
            return 1;
        }
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {

            for (int j = 1; j <= i; j++) {
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
        if (s1 == null || s2 == null || s3 == null) {
            return false;
        }
        int m = s1.length();
        int n = s2.length();
        return false;
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
        TreeNode prev = null;
        Stack<TreeNode> stack = new Stack<>();
        TreeNode node = root;
        while (!stack.isEmpty() || node != null) {
            while (node != null) {
                stack.push(node);
                node = node.left;
            }
            node = stack.pop();
            if (prev == null) {
                prev = node;
            } else {
                if (prev.val >= node.val) {
                    return false;
                }
                prev = node;
            }
            node = node.right;
        }
        return true;

    }


    /**
     * 99. Recover Binary Search Tree
     *
     * @param root
     */
    public void recoverTree(TreeNode root) {
        if (root == null) {
            return;
        }
        TreeNode first = null;
        TreeNode second = null;

        Stack<TreeNode> stack = new Stack<>();

        TreeNode node = root;

        TreeNode prev = null;

        while (!stack.isEmpty() || node != null) {
            while (node != null) {
                stack.push(node);
                node = node.left;
            }

            node = stack.pop();

            if (prev == null) {
                prev = node;
            } else {

                if (first == null && prev.val >= node.val) {
                    first = prev;
                }
                if (first != null && prev.val >= node.val) {
                    second = node;
                }
                prev = node;
            }


            node = node.right;
        }
        if (first != null && second != null) {
            int val = first.val;
            first.val = second.val;
            second.val = val;
        }

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
        return this.isSymmetricTree(root.left, root.right);
    }

    private boolean isSymmetricTree(TreeNode left, TreeNode right) {
        if (left == null && right == null) {
            return true;
        }
        if (left == null || right == null) {
            return false;
        }
        if (left.val == right.val) {
            return this.isSymmetricTree(left.left, right.right) && this.isSymmetricTree(left.right, right.left);
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
        Deque<TreeNode> deque = new LinkedList<>();
        deque.add(root);
        while (!deque.isEmpty()) {
            int size = deque.size();
            List<Integer> tmp = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                TreeNode node = deque.poll();
                tmp.add(node.val);
                if (node.left != null) {
                    deque.add(node.left);
                }
                if (node.right != null) {
                    deque.add(node.right);
                }
            }
            ans.add(tmp);
        }
        return ans;
    }


    /**
     * 189. Rotate Array
     *
     * @param nums
     * @param k
     */
    public void rotate(int[] nums, int k) {
        if (nums == null || nums.length == 0 || k < 0) {
            return;
        }

        k %= nums.length;
        this.reverseNums(nums, 0, nums.length - 1);
        this.reverseNums(nums, 0, k - 1);
        this.reverseNums(nums, k, nums.length - 1);

    }


}

