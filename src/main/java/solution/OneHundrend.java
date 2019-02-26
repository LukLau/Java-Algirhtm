package solution;


import org.dora.algorithm.datastructe.ListNode;

import java.util.*;

/**
 * @author liulu
 * @date 2019/02/16
 */
public class OneHundrend {

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
        int left = 0, longest = 0;
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
        int m = nums1.length, n = nums2.length;
        if (m > n) {
            return findMedianSortedArrays(nums2, nums1);
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
        int m = s.length(), left = 0, longest = 0;
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
        int sign = 1, idx = 0;
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
        boolean[][] dp = initDp(p, m, n);
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
        int result = 0, left = 0, right = height.length - 1;
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
            int left = i + 1, right = nums.length - 1, target = 0 - nums[i];
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
            l1.next = mergeTwoLists(l1.next, l2);
            return l1;
        } else {
            l2.next = mergeTwoLists(l1, l2.next);
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
        generateParenthesis(ans, "", 0, 0, n);
        return ans;
    }

    private void generateParenthesis(List<String> ans, String s, int open, int close, int n) {
        if (s.length() == 2 * n) {
            ans.add(s);
        }
        if (open < n) {
            generateParenthesis(ans, s + "(", open + 1, close, n);
        }
        if (close < open) {
            generateParenthesis(ans, s + ")", open, close + 1, n);
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
        ListNode newHead = reverseList(head, currNode);
        head.next = reverseKGroup(currNode, k);
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
            reverseArray(nums, index, nums.length - 1);
        } else {
            int j = nums.length - 1;
            while (j > index - 1) {
                if (nums[j] > nums[index - 1]) {
                    break;
                }
                j--;
            }
            swap(nums, index - 1, j);
            reverseArray(nums, index, nums.length - 1);
        }
    }

    private void reverseArray(int[] nums, int start, int end) {
        if (start > end) {
            return;
        }
        for (int i = start; i <= (start + end) / 2; i++) {
            swap(nums, i, start + end - i);
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
        if (isEmpty(candidates)) {
            return new ArrayList<>();
        }
        Arrays.sort(candidates);
        List<List<Integer>> ans = new ArrayList<>();
        combinationSum(ans, new ArrayList<>(), 0, candidates, target);
        return ans;
    }

    private void combinationSum(List<List<Integer>> ans, List<Integer> integers, int index, int[] candidates, int target) {
        if (target == 0) {
            ans.add(new ArrayList<>(integers));
        }
        for (int i = index; i < candidates.length && candidates[i] <= target; i++) {
            integers.add(candidates[i]);
            combinationSum(ans, integers, i, candidates, target - candidates[i]);
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
        if (isEmpty(candidates)) {
            return new ArrayList<>();
        }
        List<List<Integer>> ans = new ArrayList<>();
        Arrays.sort(candidates);
        combinationSum2(ans, new ArrayList<>(), 0, candidates, target);
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
            combinationSum2(ans, tmp, i + 1, candidates, target - candidates[i]);
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
        if (isEmpty(nums)) {
            return 1;
        }
        for (int i = 0; i < nums.length; i++) {
            while (nums[i] > 0 && nums[i] <= nums.length && nums[i] != nums[nums[i] - 1]) {
                swap(nums, i, nums[i] - 1);
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
        if (isEmpty(height)) {
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
        if (isStringEmpty(num1) || isStringEmpty(num2)) {
            return "";
        }
        int m = num1.length(), n = num2.length();
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
                    dp[i][j] = match(s, p, i, j) ? dp[i - 1][j - 1] : false;
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
        permute(ans, new ArrayList<>(), used, nums);
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
            permute(ans, tmp, used, nums);
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
        if (isEmpty(nums)) {
            return new ArrayList<>();
        }
        Arrays.sort(nums);
        List<List<Integer>> ans = new ArrayList<>();
        boolean[] used = new boolean[nums.length];
        permuteUnique(ans, new ArrayList<>(), used, nums);
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
            permuteUnique(ans, tmp, used, nums);
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
                swapMatrix(matrix, i, j);
            }
        }
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length / 2; j++) {
                swapMatrixRow(matrix, i, j);
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
        solveNQueens(ans, nQueens, 0, n);
        return ans;

    }

    private void solveNQueens(List<List<String>> ans, char[][] nQueens, int row, int n) {
        if (row == n) {
            ans.add(construct(nQueens));
        }
        for (int col = 0; col < n; col++) {
            if (!checkExist(nQueens, col, row, n)) {
                nQueens[row][col] = 'Q';
                solveNQueens(ans, nQueens, row + 1, n);
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
        return totalNQueens(dp, 0, n);
    }

    private int totalNQueens(int[] dp, int row, int n) {
        int result = 0;
        if (row == n) {
            result++;
            return result;
        }
        for (int col = 0; col < n; col++) {
            if (!checkExist(dp, row, col)) {
                dp[row] = col;
                result += totalNQueens(dp, row + 1, n);
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
        int row = matrix.length;
        int col = matrix[0].length;
        int i = 0;
        int j = 0;
        while (i < row && j < col) {
            for (int k = j; k < col; k++) {
                ans.add(matrix[i][k]);
            }
            for (int k = i + 1; k < row; k++) {
                ans.add(matrix[k][j]);
            }
            if (i != j) {
                for (int k = col - 1; k >= 0; k--) {
                    ans.add(matrix[])
                }
            }

        }
    }
}
