package org.learn.algorithm.leetcode;

import org.learn.algorithm.datastructure.ListNode;

import java.util.*;

/**
 * @author luk
 * @date 2021/4/6
 */
public class FirstPage {

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
        int[] result = new int[2];
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(target - nums[i])) {
                result[0] = map.get(target - nums[i]);
                result[1] = i;
                return result;
            }
            map.put(nums[i], i);
        }
        return result;
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
            int val = (l1 == null ? 0 : l1.val) + (l2 == null ? 0 : l2.val) + carry;

            dummy.next = new ListNode(val % 10);

            carry = val / 10;

            dummy = dummy.next;
            l1 = l1 == null ? null : l1.next;
            l2 = l2 == null ? null : l2.next;
        }
        return root.next;
    }

    public int reverse(int x) {
        int result = 0;
        while (x != 0) {
            if (result > Integer.MAX_VALUE / 10 || result < Integer.MIN_VALUE / 10) {
                return 0;
            }
            result = result * 10 + x % 10;

            x /= 10;
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

        int len = digits.length();

        LinkedList<String> deque = new LinkedList<>();

        deque.offer("");
        for (int i = 0; i < len; i++) {
            int index = Character.getNumericValue(digits.charAt(i));

            String word = map[index];

            while (deque.peek().length() == i) {
                String poll = deque.poll();

                char[] words = word.toCharArray();

                for (char t : words) {
                    deque.offer(poll + t);
                }
            }
        }
        return deque;
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
        char[] words = str.toCharArray();
        int index = 0;
        int sign = 1;
        if (words[index] == '+' || words[index] == '-') {
            sign = words[index] == '+' ? 1 : -1;
            index++;
        }
        long result = 0;
        while (index < words.length && Character.isDigit(words[index])) {
            result = result * 10 + Character.getNumericValue(words[index++]);
            if (result > Integer.MAX_VALUE) {
                return sign == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
            }
        }
        return (int) (result * sign);
    }


    public int maxArea(int[] height) {
        if (height == null || height.length == 0) {
            return 0;
        }
        int left = 0;
        int right = height.length - 1;
        int leftEdge = 0;
        int rightEdge = 0;
        int result = 0;
        while (left < right) {
            if (height[left] <= height[right]) {
                if (leftEdge <= height[left]) {
                    leftEdge = height[left];
                } else {
                    result += leftEdge - height[left];
                }
                left++;
            } else {
                if (rightEdge <= height[right]) {
                    rightEdge = height[right];
                } else {
                    result += rightEdge - height[right];
                }
                right--;
            }
        }
        return result;
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
        Stack<Integer> stack = new Stack<>();
        char[] words = s.toCharArray();
        int result = 0;
        int left = -1;
        for (int i = 0; i < words.length; i++) {
            if (words[i] == '(') {
                stack.push(i);
            } else {
                if (!stack.isEmpty() && words[stack.peek()] == '(') {
                    stack.pop();
                } else {
                    left = i;
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

    public int longestValidParenthesesII(String s) {
        if (s == null) {
            return 0;
        }
        Stack<Integer> stack = new Stack<>();
        char[] words = s.toCharArray();
        for (int i = 0; i < words.length; i++) {
            if (words[i] == '(' || stack.isEmpty()) {
                stack.push(i);
            } else {
                if (words[stack.peek()] == '(') {
                    stack.pop();
                } else {
                    stack.push(i);
                }

            }
        }
        if (stack.isEmpty()) {
            return words.length;
        }
        int result = 0;
        int rightEdge = s.length();
        while (!stack.isEmpty()) {
            Integer left = stack.pop();
            result = Math.max(result, rightEdge - left - 1);
            rightEdge = left;
        }
        result = Math.max(result, rightEdge);
        return result;
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
     * 41. First Missing Positive
     *
     * @param nums
     * @return
     */
    public int firstMissingPositive(int[] nums) {
        for (int i = 0; i < nums.length; i++) {
            while (nums[i] >= 1 && nums[i] <= nums.length && nums[i] != nums[nums[i] - 1]) {
                swapValue(nums, i, nums[i] - 1);
            }
        }
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != i + 1) {
                return i + 1;
            }
        }
        return nums.length + 1;
    }

    private void swapValue(int[] tmp, int i, int j) {
        int val = tmp[i];
        tmp[i] = tmp[j];
        tmp[j] = val;
    }

    /**
     * 43. Multiply Strings
     *
     * @param num1
     * @param num2
     * @return
     */
    public String multiply(String num1, String num2) {
        if (num1 == null || num2 == null) {
            return "";
        }
        int m = num1.length();
        int n = num2.length();
        int[] nums = new int[m + n];
        for (int i = m - 1; i >= 0; i--) {
            for (int j = n - 1; j >= 0; j--) {
                int val = Character.getNumericValue(num1.charAt(i)) * Character.getNumericValue(num2.charAt(j)) + nums[i + j + 1];

                nums[i + j + 1] = val % 10;

                nums[i + j] += val / 10;

            }
        }
        StringBuilder builder = new StringBuilder();
        for (int num : nums) {
            if (!(num == 0 && builder.length() == 0)) {
                builder.append(num);
            }
        }
        return builder.length() == 0 ? "0" : builder.toString();
    }

    /**
     * 49. Group Anagrams
     *
     * @param strs
     * @return
     */
    public List<List<String>> groupAnagrams(String[] strs) {
        if (strs == null) {
            return new ArrayList<>();
        }
        Map<String, List<String>> map = new HashMap<>();
        for (String str : strs) {
            char[] words = str.toCharArray();

            Arrays.sort(words);

            String key = String.valueOf(words);

            List<String> list = map.getOrDefault(key, new ArrayList<>());
            list.add(str);

            map.put(key, list);

        }
        return new ArrayList<>(map.values());
    }


    /**
     * 56. Merge Intervals
     *
     * @param intervals
     * @return
     */
    public int[][] merge(int[][] intervals) {
        if (intervals == null || intervals.length == 0) {
            return new int[][]{};
        }
        PriorityQueue<int[]> queue = new PriorityQueue<>(intervals.length, Comparator.comparingInt(o -> o[0]));
        for (int[] interval : intervals) {
            queue.offer(interval);
        }
        LinkedList<int[]> linkedList = new LinkedList<>();
        while (!queue.isEmpty()) {
            int[] poll = queue.poll();
            if (linkedList.isEmpty() || linkedList.peekLast()[1] < poll[0]) {
                linkedList.add(poll);
            } else {
                int[] peek = linkedList.peekLast();
                peek[0] = Math.min(peek[0], poll[0]);
                peek[1] = Math.max(peek[1], poll[1]);
            }
        }
        return linkedList.toArray(new int[][]{});
    }


    /**
     * 57. Insert Interval
     *
     * @param intervals
     * @param newInterval
     * @return
     */
    public int[][] insert(int[][] intervals, int[] newInterval) {
        if (intervals == null) {
            return new int[][]{};
        }
        Arrays.sort(intervals, Comparator.comparingInt(o -> o[0]));

        LinkedList<int[]> deque = new LinkedList<>();

        int index = 0;
        while (index < intervals.length && intervals[index][1] < newInterval[0]) {
            deque.offer(intervals[index++]);
        }
        while (index < intervals.length && intervals[index][0] <= newInterval[1]) {
            newInterval[0] = Math.min(newInterval[0], intervals[index][0]);
            newInterval[1] = Math.max(newInterval[1], intervals[index][1]);
            index++;
        }
        deque.offer(newInterval);
        while (index < intervals.length) {
            deque.offer(intervals[index++]);
        }
        return deque.toArray(new int[][]{});
    }

    /**
     * 71. Simplify Path
     *
     * @param path
     * @return
     */
    public String simplifyPath(String path) {
        if (path == null || path.isEmpty()) {
            return "/";
        }
        String[] words = path.split("/");
        LinkedList<String> deque = new LinkedList<>();
        for (String word : words) {
            if (word.isEmpty() || ".".endsWith(word)) {
                continue;
            }
            if ("..".equals(word)) {
                deque.pollLast();
            } else {
                deque.offer(word);
            }
        }

        String tmp = "";
        for (String s : deque) {
            tmp = tmp + "/" + s;
        }
        return tmp.isEmpty() ? "/" : tmp;

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
        int blue = nums.length - 1;
        int zero = 0;
        int index = 0;
        while (index < nums.length) {
            while (index < blue && nums[index] == 2) {
                swapValue(nums, index, blue--);
            }
            while (index > zero && nums[index] == 0) {
                swapValue(nums, index, zero++);
            }
            index++;
        }
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
            return deleteDuplicates(node);
        }
        head.next = deleteDuplicates(head.next);
        return head;
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
        int result = 0;
        for (int i = 0; i <= heights.length; i++) {
            int h = i == heights.length ? 0 : heights[i];
            if (stack.isEmpty() || heights[stack.peek()] <= h) {
                stack.push(i);
            } else {
                Integer pop = stack.pop();
                int side = stack.isEmpty() ? i : i - stack.peek() - 1;

                result = Math.max(result, side * heights[pop]);
                i--;
            }
        }
        return result;
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
        int k = m + n;
        m--;
        n--;
        k--;
        while (m >= 0 && n >= 0) {
            if (nums1[m] < nums2[n]) {
                nums1[k--] = nums2[n--];
            } else {
                nums1[k--] = nums1[m--];
            }
        }
        while (n >= 0) {
            nums1[k--] = nums2[n--];
        }
    }


    /**
     * 93. Restore IP Addresses
     *
     * @param s
     * @return
     */
    public List<String> restoreIpAddresses(String s) {
        if (s == null || s.isEmpty()) {
            return new ArrayList<>();
        }
        List<String> result = new ArrayList<>();
        int len = s.length();
        for (int i = 1; i <= 3 && i <= len - 3; i++) {
            for (int j = i + 1; j <= i + 3 && j <= len - 2; j++) {
                for (int k = j + 1; k < k + 3 && k <= len - 1; k++) {
                    String a = s.substring(0, i);
                    String b = s.substring(i, j);
                    String c = s.substring(j, k);
                    String d = s.substring(k);
                    if (validIpSeq(a) && validIpSeq(b) && validIpSeq(c) && validIpSeq(d)) {
                        String tmp = a + "." + b + "." + c + "." + d;
                        result.add(tmp);
                    }
                }
            }
        }
        return result;
    }

    private boolean validIpSeq(String s) {
        if (s.isEmpty()) {
            return false;
        }

        int len = s.length();
        if (len > 3) {
            return false;
        }
        int num = Integer.parseInt(s);
        if (!(num >= 0 && num <= 255)) {
            return false;
        }
        return len <= 1 || s.charAt(0) != '0';
    }

    public static void main(String[] args) {
        FirstPage page = new FirstPage();
        List<String> result = page.restoreIpAddresses("010010");
        System.out.println(result);
    }


}
