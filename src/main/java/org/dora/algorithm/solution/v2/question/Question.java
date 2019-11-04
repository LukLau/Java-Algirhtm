package org.dora.algorithm.solution.v2.question;

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
        question.divide(Integer.MIN_VALUE, 1);
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


    /**
     * 29. Divide Two Integers
     *
     * @param dividend
     * @param divisor
     * @return
     */
    public int divide(int dividend, int divisor) {
        int sign = 1;
        boolean correct = (dividend > 0 && divisor > 0)
                || (dividend < 0 && divisor < 0);
        if (!correct) {
            sign = -1;
        }
        long dvd = Math.abs((long) dividend);
        long dvs = Math.abs((long) divisor);

        if (dvs == 0) {
            return Integer.MAX_VALUE;
        }

        if (dvd == 0 || dvd < dvs) {
            return 0;
        }

        long sum = this.getDividend(dvd, dvs);

        if (sum > Integer.MAX_VALUE) {
            return sign == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
        }
        return (int) sum * sign;

    }

    private long getDividend(long dvd, long dvs) {
        long result = 0;

        while (dvd >= dvs) {
            long num = 1;

            long tmp = dvs;

            while (dvd >= (tmp << 1)) {

                tmp <<= 1;

                num <<= 1;
            }
            dvd -= tmp;

            result += num;
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
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') {
                stack.push(i);
            } else if (!stack.isEmpty() && s.charAt(stack.peek()) == '(') {
                stack.pop();
            } else {
                stack.push(i);
            }
        }
        if (stack.isEmpty()) {
            return s.length();
        }
        int result = 0;

        int now = s.length();

        while (!stack.isEmpty()) {

            int index = stack.pop();

            result = Math.max(result, now - 1 - index);

            now = index;
        }

        result = Math.max(result, now);

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
            }
            if (nums[left] <= nums[mid]) {
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
     *
     * @param nums
     * @param target
     * @return
     */
    public int[] searchRange(int[] nums, int target) {
        int[] ans = new int[]{-1, -1};
        if (nums == null || nums.length == 0) {
            return ans;
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
        if (nums[left] != target) {
            return ans;
        }
        ans[0] = left;
        right = nums.length - 1;
        while (left < right) {
            int mid = (left + right) / 2 + 1;
            if (nums[mid] > target) {
                right = mid - 1;
            } else {
                left = mid;
            }
        }
        ans[1] = left;
        return ans;
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

    private void swap(int[] nums, int i, int j) {
        int tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;
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

        int leftEdge = 0;
        int rightEdge = 0;

        int result = 0;

        while (left < right) {
            if (height[left] <= height[right]) {
                if (height[left] >= leftEdge) {
                    leftEdge = height[left];
                } else {
                    result += leftEdge - height[left];
                }
                left++;
            } else {
                if (height[right] >= rightEdge) {
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

                int value = (Character.getNumericValue(num1.charAt(i))
                        * Character.getNumericValue(num2.charAt(j)))
                        + pos[i + j + 1];

                pos[i + j] += value / 10;

                pos[i + j + 1] = value % 10;
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
     * 44. Wildcard Matching
     *
     * @param s
     * @param p
     * @return
     */
    public boolean isMatchII(String s, String p) {
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
            dp[0][j] = p.charAt(j - 1) == '*' && dp[0][j - 1];
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

    /**
     * 45. Jump Game II
     *
     * @param nums
     * @return
     * @tag
     */
    public int jump(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int step = 0;
        int current = 0;
        int farthest = 0;
        for (int i = 0; i < nums.length - 1; i++) {
            farthest = Math.max(i + nums[i], farthest);

            if (i == current) {

                step++;

                current = farthest;
            }
        }
        return step;
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
        HashMap<String, List<String>> keyMap = new HashMap<>();
        for (String str : strs) {
            char[] chars = str.toCharArray();
            Arrays.sort(chars);

            String key = String.valueOf(chars);

            List<String> list = keyMap.getOrDefault(key, new ArrayList<>());

            list.add(str);

            keyMap.put(key, list);
        }
        return new ArrayList<>(keyMap.values());
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
        int result = Integer.MIN_VALUE;
        int local = 0;
        for (int num : nums) {
            local = local >= 0 ? local + num : num;
            result = Math.max(result, local);
        }
        return result;
    }


    /**
     * 贪心算法
     * 55. Jump Game
     *
     * @param nums
     * @return
     */
    public boolean canJump(int[] nums) {
        if (nums == null || nums.length == 0) {
            return false;
        }
        int furthest = 0;
        for (int i = 0; i < nums.length - 1 && i <= furthest; i++) {
            furthest = Math.max(i + nums[i], furthest);
        }
        return furthest >= nums.length - 1;
    }


    /**
     * todo 巍
     * 56. Merge Intervals
     *
     * @param intervals
     * @return
     */
    public int[][] merge(int[][] intervals) {
        if (intervals == null || intervals.length == 0) {
            return new int[][]{};
        }
        return null;
    }

    /**
     * 61. Rotate List
     *
     * @param head
     * @param k
     * @return
     */
    public ListNode rotateRight(ListNode head, int k) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode current = head;
        int count = 1;
        while (current.next != null) {
            current = current.next;
            count++;
        }
        current.next = head;
        ListNode slow = head;
        if ((k %= count) != 0) {
            for (int i = 0; i < count - k; i++) {

                slow = slow.next;

                current = current.next;
            }
        }
        current.next = null;

        return slow;

    }

    /**
     * 65. Valid Number
     *
     * @param s
     * @return
     */
    public boolean isNumber(String s) {
        if (s == null) {
            return false;
        }
        s = s.trim();

        if (s.length() == 0) {
            return false;
        }
        boolean existSign = false;
        boolean existDit = false;
        boolean existE = false;

        for (int i = 0; i < s.length(); i++) {
            char value = s.charAt(i);

        }
        return false;

    }


    /**
     * todo 第二种解法
     * 68. Text Justification
     *
     * @param words
     * @param maxWidth
     * @return
     */
    public List<String> fullJustify(String[] words, int maxWidth) {
        if (words == null || words.length == 0 || maxWidth <= 0) {
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
            StringBuilder builder = new StringBuilder();

            int lineLen = maxWidth - line + 1;

            int wordNum = endIndex - startIndex;

            boolean lastRow = endIndex == words.length;

            if (endIndex - startIndex == 1) {
                builder.append(words[startIndex]);
            } else {

                int countOfWord = lastRow ? 1 : lineLen / (wordNum - 1) + 1;

                int extraOfWord = lastRow ? 0 : lineLen % (wordNum - 1);

                builder.append(this.constructCurrentLine(words, startIndex, endIndex, countOfWord, extraOfWord));
            }
            String str = this.fillLen(builder.toString(), maxWidth);

            ans.add(str);

            startIndex = endIndex;
        }
        return ans;
    }

    private String constructCurrentLine(String[] words, int startIndex, int endIndex,
                                        int countOfWord, int extraOfWord) {
        StringBuilder builder = new StringBuilder();

        for (int i = startIndex; i < endIndex; i++) {
            builder.append(words[i]);
            int tmp = countOfWord;

            while (tmp-- > 0) {

                builder.append(" ");
            }
            if (extraOfWord-- > 0) {
                builder.append(" ");
            }
        }
        return builder.toString();
    }


    private String fillLen(String str, int maxWidth) {
        while (str.length() < maxWidth) {
            str = str + " ";
        }
        while (str.length() > maxWidth) {
            str = str.substring(0, str.length() - 1);
        }
        return str;
    }

    /**
     * 80. Remove Duplicates from Sorted Array II
     *
     * @param nums
     * @return
     */
    public int removeDuplicates(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int count = 1;
        int index = 1;
        for (int i = 1; i < nums.length; i++) {
            if (count >= 3) {
                continue;
            }
            if (nums[i] == nums[i - 1]) {
                index++;
                count++;
            } else {
                count = 1;
                nums[index++] = nums[i];
            }
        }
        return index;
    }


}

