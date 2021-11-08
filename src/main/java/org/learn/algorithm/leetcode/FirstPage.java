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
        LinkedList<String> linkedList = new LinkedList<>();
        linkedList.offer("");
        int len = digits.length();
        for (int i = 0; i < len; i++) {
            char digit = digits.charAt(i);
            int index = Character.getNumericValue(digit);
            String word = map[index];
            while (linkedList.peekFirst().length() == i) {
                char[] words = word.toCharArray();
                String prefix = linkedList.pollFirst();
                for (char tmp : words) {
                    linkedList.offer(prefix + tmp);

                }
            }
        }
        return linkedList;
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
     * 12. Integer to Roman
     *
     * @param num
     * @return
     */
    public String intToRoman(int num) {
        String[] one = new String[]{"", "I", "IStringSolution.I", "III", "IV", "V", "VI", "VII", "VIII", "IX"};
        String[] two = new String[]{"", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"};
        String[] three = new String[]{"", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"};
        String[] four = new String[]{"", "M", "MM", "MMM"};
        return four[num / 1000] + three[num % 1000 / 100] + two[num % 100 / 10] + one[num % 10];
    }

    public int romanToInt(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        }
        Map<String, Integer> map = new HashMap<>();
        map.put("I", 1);
        map.put("V", 5);
        map.put("X", 10);
        map.put("L", 50);
        map.put("C", 100);
        map.put("D", 500);
        map.put("M", 1000);
        char[] words = s.toCharArray();
        StringBuilder builder = new StringBuilder();
        for (char word : words) {
            builder.append(map.get(word + ""));
        }
        int result = 0;
        char[] chars = builder.toString().toCharArray();
        for (int i = 0; i < chars.length; i++) {
            result += Character.getNumericValue(chars[i]);
            if (i > 0 && (Character.getNumericValue(chars[i]) > Character.getNumericValue(chars[i - 1]))) {
                result -= 2 * Character.getNumericValue(chars[i - 1]);
            }
        }
        return result;

    }


    public int threeSumClosest(int[] nums, int target) {
        if (nums == null || nums.length <= 2) {
            return -1;
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
                int val = nums[i] + nums[left] + nums[right];
                if (val == target) {
                    return target;
                } else if (val < target) {
                    left++;
                } else {
                    right--;
                }
                if (Math.abs(result - target) > Math.abs(val - target)) {
                    result = val;
                }
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

        int left = -1;

        int result = 0;

        char[] words = s.toCharArray();

        for (int i = 0; i < words.length; i++) {
            char tmp = words[i];
            if (tmp == '(') {
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

        int m = s.length();

        for (int i = 0; i < words.length; i++) {
            char tmp = words[i];
            if (stack.isEmpty() || tmp == '(') {
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
            return m;
        }
        int a = m;
        int result = 0;
        while (!stack.isEmpty()) {
            int b = stack.pop();

            result = Math.max(result, a - b - 1);

            a = b;
        }
        result = Math.max(result, a);

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
        int leftEdge = 0;

        int rightEdge = 0;

        int left = 0;
        int right = height.length - 1;

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
     * 41. First Missing Positive
     *
     * @param nums
     * @return
     */
    public int firstMissingPositive(int[] nums) {
        if (nums == null || nums.length == 0) {
            return -1;
        }
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
        int[] pos = new int[m + n];
        for (int i = m - 1; i >= 0; i--) {
            for (int j = n - 1; j >= 0; j--) {
                int val = Character.getNumericValue(num1.charAt(i)) * Character.getNumericValue(num2.charAt(j)) + pos[i + j + 1];

                pos[i + j + 1] = val % 10;

                pos[i + j] += val / 10;
            }
        }
        StringBuilder builder = new StringBuilder();
        for (int num : pos) {
            if (!(builder.length() == 0 && num == 0)) {
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
        Arrays.sort(intervals, Comparator.comparingInt(o -> o[0]));
        List<int[]> result = new ArrayList<>();
        for (int[] interval : intervals) {
            if (result.isEmpty() || result.get(result.size() - 1)[1] < interval[0]) {
                result.add(interval);
            } else {
                int[] pre = result.get(result.size() - 1);
                pre[0] = Math.min(pre[0], interval[0]);
                pre[1] = Math.max(pre[1], interval[1]);
            }
        }
        return result.toArray(new int[][]{});
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
        LinkedList<String> linkedList = new LinkedList<>();

        for (String word : words) {
            if ("..".equals(word)) {
                if (!linkedList.isEmpty()) {
                    linkedList.pollLast();
                }
            } else if (!"".equals(word) && !".".equals(word)) {
                linkedList.offer(word);
            }
        }
        StringBuilder s = new StringBuilder();

        for (String tmp : linkedList) {
            s.append("/").append(tmp);
        }
        return s.length() == 0 ? "/" : s.toString();
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

        int red = 0;
        for (int i = 0; i < nums.length; i++) {
            while (i < blue && nums[i] == 2) {
                swapValue(nums, i, blue--);
            }
            while (i > red && nums[i] == 0) {
                swapValue(nums, i, red++);
            }
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
            int side = i == heights.length ? 0 : heights[i];
            if (stack.isEmpty() || heights[stack.peek()] <= side) {
                stack.push(i);
            } else {
                Integer pop = stack.pop();
                int edge = stack.isEmpty() ? i : i - stack.peek() - 1;
                result = Math.max(result, heights[pop] * edge);
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
        for (int i = 1; i < 4 && i < len - 2; i++) {
            for (int j = i + 1; j < i + 4 && j < len - 1; j++) {
                for (int k = j + 1; k < j + 4 && k < len; k++) {
                    String a = s.substring(0, i);
                    String b = s.substring(i, j);
                    String c = s.substring(j, k);
                    String d = s.substring(k);
                    if (validIpSeq(a) && validIpSeq(b) && validIpSeq(c) && validIpSeq(d)) {
                        result.add(a + "." + b + "." + c + "." + d);
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
        int length = s.length();
        if (length > 3) {
            return false;
        }
        int val = Integer.parseInt(s);
        if (val < 0 || val > 255) {
            return false;
        }
        return length < 2 || s.charAt(0) != '0';
    }

    public static void main(String[] args) {
        FirstPage page = new FirstPage();

//        System.out.println(page.intToRoman(3));
//        System.out.println(page.intToRoman(4));
//        System.out.println(page.intToRoman(9));

        page.romanToInt("IX");
    }


}
