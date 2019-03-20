package org.dora.algorithm.question;

import org.dora.algorithm.datastructe.ListNode;

import java.util.*;

/**
 * @author liulu
 * @date 2019-03-17
 */
public class Question {

    /**
     * 52. N-Queens II
     *
     * @param n
     * @return
     */
    private int count = 0;

    public static void main(String[] args) {
        Question question = new Question();
        String[] words = new String[]{"abcdefg", "qwertyui", "yuiopoppp"};
        question.simplifyPath("/home//foo/");

    }

    /**
     * 4. Median of Two Sorted Arrays
     *
     * @param nums1
     * @param nums2
     * @return
     */
    // todo 不懂
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int m = nums1.length;
        int n = nums2.length;
        if (m > n) {
            return findMedianSortedArrays(nums2, nums1);
        }
        int imin = 0;
        int imax = nums1.length;
        int max_left = 0;
        int min_right = 0;
        while (imin <= imax) {
            int i = (imin + imax) / 2;
            int j = (m + n + 1) / 2 - i;

            if (i < imax && nums1[i] < nums2[j - 1]) {
                imin = i + 1;
            } else if (i > 0 && nums2[i - 1] > nums2[j]) {
                imax = i - 1;
            } else {
                if (i == 0) {
                    max_left = nums2[j - 1];
                } else if (j == 0) {
                    max_left = nums1[i - 1];
                } else {
                    max_left = Math.max(nums1[i - 1], nums2[j - 1]);
                }
                if ((m + n) % 2 != 0) {
                    return max_left;
                }
                if (i == m) {
                    min_right = nums2[j];
                } else if (j == n) {
                    min_right = nums1[i];
                } else {
                    min_right = Math.min(nums2[j], nums1[i]);
                }

                return (max_left + min_right) / 2.0;
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

        boolean[][] dp = new boolean[m][m];

        int result = 0;

        int left = 0;
        for (int i = 0; i < s.length(); i++) {
            for (int j = 0; j < i; j++) {
                if (i - j <= 2) {
                    dp[i][j] = s.charAt(i) == s.charAt(j);
                } else {
                    dp[i][j] = (s.charAt(j) == s.charAt(i) && dp[i - 1][j + 1]);
                }
                if (dp[i][j] && i - j + 1 > result) {
                    left = j;
                    result = i - j + 1;
                }
            }
        }
        if (result > 0) {
            return s.substring(left, left + result);
        }
        return s;
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


        boolean[][] dp = new boolean[m + 1][n + 1];

        dp[0][0] = true;
        for (int j = 1; j <= n; j++) {
            dp[0][j] = p.charAt(j - 1) == '*' && dp[0][j - 2];
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
                        dp[i][j] = dp[i][j - 1] || dp[i - 1][j] || dp[i][j - 2];
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
        int left = 0;
        int right = height.length - 1;
        int result = 0;
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
     * @param digits
     * @return
     */
    public List<String> letterCombinations(String digits) {
        if (digits == null || digits.length() == 0) {
            return new ArrayList<>();
        }
        LinkedList<String> ans = new LinkedList<>();
        String[] map = new String[]{"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        ans.add("");
        for (int i = 0; i < digits.length(); i++) {
            int index = digits.charAt(i) - '0';
            String value = map[index];

            while (ans.peek().length() == i) {
                String str = ans.poll();
                for (char c : value.toCharArray()) {
                    ans.add(str + c);
                }

            }
        }
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
        PriorityQueue<ListNode> priorityQueue = new PriorityQueue<>(Comparator.comparing(o -> o.val));
        for (ListNode listNode : lists) {
            if (listNode != null) {
                priorityQueue.add(listNode);
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

            slow.next = fast.next;

            fast.next = slow;

            dummy.next = fast;

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
        if (head == null) {
            return null;
        }
        ListNode currNode = head;
        for (int i = 0; i < k; i++) {
            if (currNode == null) {
                return head;
            }
            currNode = currNode.next;
        }
        ListNode newHead = reverseListNode(head, currNode);
        head.next = reverseKGroup(currNode, k);
        return newHead;
    }

    private ListNode reverseListNode(ListNode first, ListNode last) {
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
     * 29. Divide Two Integers
     *
     * @param dividend
     * @param divisor
     * @return
     */
    // todo 一直未搞懂
    public int divide(int dividend, int divisor) {
        if (divisor == 0) {
            return Integer.MAX_VALUE;
        }
        if (dividend == Integer.MIN_VALUE && divisor == -1) {
            return Integer.MAX_VALUE;
        }
        int sign = (dividend < 0) ^ (divisor < 0) ? -1 : 1;
        int dvd = Math.abs(dividend);
        int dvs = Math.abs(divisor);

        long result = 0;
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
        return (int) result * sign;
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
            if (nums[index - 1] < nums[index]) {
                break;
            }
            index--;
        }
        if (index == 0) {
            reverseNums(nums, 0, nums.length - 1);
        } else {
            int j = nums.length - 1;
            while (j > index - 1) {
                if (nums[j] > nums[index - 1]) {
                    break;
                }
                j--;
            }
            swap(nums, index - 1, j);
            reverseNums(nums, index, nums.length - 1);
        }
    }

    private void reverseNums(int[] nums, int start, int end) {
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
        Stack<Integer> stack = new Stack<>();
        int left = 0;
        int result = 0;
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') {
                stack.push(i);
            } else {
                if (!stack.isEmpty() && s.charAt(stack.peek()) == '(') {
                    stack.pop();
                    if (stack.isEmpty()) {
                        result = Math.max(result, i - left);
                    } else {
                        result = Math.max(result, i - stack.peek());
                    }
                } else {
                    left = i;
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
        while (left < right) {
            int mid = (left + right) / 2;
            if (nums[mid] == target) {
                return mid;
            }
            if (nums[left] <= nums[mid]) {
                if (nums[left] <= target && target < nums[mid]) {
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
     * 34. Find First and Last Position of Element in Sorted Array
     *
     * @param nums
     * @param target
     * @return
     */
    public int[] searchRange(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return new int[2];
        }
        int left = 0;
        int right = nums.length - 1;
        int[] ans = new int[2];
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < target) {
                right = mid + 1;
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
            int mid = left + (right - left) / 2;
            if (nums[mid] < target) {
                right = mid + 1;
            } else {
                right = mid;
            }
        }
        ans[1] = right;
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
            return -1;
        }
        int left = 0;
        int right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) >> 1;
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
            while (nums[i] > 0 && nums[i] <= nums.length && nums[nums[i] - 1] != nums[i]) {
                swap(nums, nums[i - 1], i);
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
        if (height == null || height.length == 0) {
            return 0;
        }
        int left = 0;
        int right = height.length - 1;
        int result = 0;
        int maxLeft = 0;
        int maxRight = 0;
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
        dp[m][n] = true;
        for (int j = n - 1; j >= 0; j--) {
            if (p.charAt(j) != '*') {
                break;
            }
            dp[m][j] = true;
        }
        for (int i = m - 1; i >= 0; i--) {
            for (int j = n - 1; j >= 0; j--) {
                if (s.charAt(i) == p.charAt(j) || p.charAt(j) == '?') {
                    dp[i][j] = dp[i + 1][j + 1];
                } else if (p.charAt(j) == '*') {
                    dp[i][j] = dp[i][j + 1] || dp[i + 1][j];
                } else {
                    dp[i][j] = false;
                }
            }
        }
        return dp[0][0];
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
        int furthest = 0;
        int curr = 0;
        for (int i = 0; i < nums.length - 1; i++) {
            furthest = Math.max(curr, i + nums[i]);
            if (i == curr) {
                step++;
                curr = furthest;
            }
        }
        return step;
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
            n = -n;
            x = 1 / x;
        }
        if (x > Integer.MAX_VALUE || x < Integer.MIN_VALUE) {
            return 0;
        }
        return n % 2 == 0 ? myPow(x * x, n / 2) : x * myPow(x * x, n / 2);
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
        solveNQueens(ans, nQueens, 0, n);
        return ans;
    }

    private void solveNQueens(List<List<String>> ans, char[][] nQueens, int row, int n) {
        if (row == n) {

            List<String> queens = construct(nQueens);

            ans.add(queens);

        }
        for (int col = 0; col < n; col++) {
            if (matchQueens(nQueens, col, row, n)) {
                nQueens[row][col] = 'Q';
                solveNQueens(ans, nQueens, row + 1, n);
                nQueens[row][col] = '.';
            }
        }
    }

    private boolean matchQueens(char[][] nQueens, int col, int row, int n) {
        for (int i = row - 1; i >= 0; i--) {
            if (nQueens[i][col] == 'Q') {
                return false;
            }
        }
        for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--) {
            if (nQueens[i][j] == 'Q') {
                return false;
            }
        }
        for (int i = row - 1, j = col + 1; i >= 0 && j < n; i--, j++) {
            if (nQueens[i][j] == 'Q') {
                return false;
            }
        }
        return true;
    }

    private List<String> construct(char[][] nQueens) {
        List<String> ans = new ArrayList<>();
        for (char[] row : nQueens) {
            ans.add(String.valueOf(row));
        }
        return ans;
    }

    public int totalNQueens(int n) {
        if (n <= 0) {
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
            if (matchTotalNQueens(dp, col, row, n)) {
                dp[row] = col;
                result += totalNQueens(dp, row + 1, n);
                dp[row] = -1;
            }
        }
        return result;
    }

    private boolean matchTotalNQueens(int[] dp, int col, int row, int n) {
        for (int i = row - 1; i >= 0; i--) {
            if (dp[i] == col || Math.abs(row - i) == Math.abs(dp[i] - col)) {
                return false;
            }
        }
        return true;
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
            bottom--;
            top++;
        }
        return ans;
    }

    /**
     * 59. Spiral Matrix II
     *
     * @param n
     * @return
     */
    public int[][] generateMatrix(int n) {
        if (n <= 0) {
            return new int[0][0];
        }
        int[][] matrix = new int[n][n];
        int left = 0;
        int right = n - 1;
        int top = 0;
        int bottom = n - 1;
        int total = 0;
        while (total != n * n) {
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
        int[] position = new int[n + 1];
        position[0] = 1;
        int factory = 1;
        for (int i = 1; i <= n; i++) {
            factory = factory * i;
            position[i] = factory;
        }
        List<Integer> ans = new ArrayList<>();
        for (int i = 1; i <= n; i++) {
            ans.add(i);
        }
        StringBuilder stringBuilder = new StringBuilder();
        k--;
        for (int i = 0; i < n; i++) {
            int index = k / position[n - 1 - i];
            stringBuilder.append(ans.get(index));
            ans.remove(index);
            k -= index * position[n - 1 - i];
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
        if (head == null) {
            return null;
        }
        ListNode fast = head;
        ListNode slow = head;
        int count = 1;
        while (fast.next != null) {
            fast = fast.next;
            count++;
        }
        k %= count;
        fast.next = head;
        for (int i = 0; i < count - k; i++) {
            fast = fast.next;
            slow = slow.next;
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
        int[] dp = new int[obstacleGrid[0].length];
        dp[0] = 1;
        for (int i = 0; i < obstacleGrid.length; i++) {
            for (int j = 0; j < obstacleGrid[0].length; j++) {
                if (obstacleGrid[i][j] == 1) {
                    dp[j] = 0;
                } else {
                    dp[j] = dp[j] + (j > 0 ? dp[j - 1] : 0);
                }
            }
        }
        return dp[obstacleGrid[0].length - 1];
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
        int m = grid.length;
        int n = grid[0].length;
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
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
        return dp[m - 1][n - 1];
    }

    /**
     * 66. Plus One
     *
     * @param digits
     * @return
     */
    public int[] plusOne(int[] digits) {
        if (digits == null || digits.length == 0) {
            return new int[0];
        }
        for (int i = digits.length - 1; i >= 0; i--) {
            if (digits[i] == 9) {
                digits[i] = 0;
            } else {
                digits[i]++;
                return digits;
            }
        }
        int[] ans = new int[digits.length + 1];
        ans[0] = 1;
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
            boolean isLast = endIndex == words.length;
            int countOfWord = endIndex - startIndex;
            StringBuilder stringBuilder = new StringBuilder();
            if (countOfWord == 1) {
                stringBuilder.append(words[startIndex]);
            } else {
                int blankWord = isLast ? 1 : 1 + (maxWidth - line + 1) / (countOfWord - 1);
                int extraWord = isLast ? 0 : (maxWidth - line + 1) % (countOfWord - 1);
                construct(words, stringBuilder, blankWord, extraWord, startIndex, endIndex);
            }
            ans.add(adjust(stringBuilder, maxWidth));
            startIndex = endIndex;
        }
        return ans;
    }

    private void construct(String[] words, StringBuilder stringBuilder, int blankWord, int extraWord, int startIndex, int endIndex) {
        for (int i = startIndex; i < endIndex; i++) {
            stringBuilder.append(words[i]);
            int tmp = blankWord;
            while (tmp-- > 0) {
                stringBuilder.append(" ");
            }
            if (extraWord-- > 0) {
                stringBuilder.append(" ");
            }
        }
    }


    private String adjust(StringBuilder sb, int maxWidth) {
        String stringBuilder = sb.toString();

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
        Stack<String> stack = new Stack<>();
        String[] strings = path.split("/");
        Set<String> skip = new HashSet<>(Arrays.asList("..", ".", ""));
        for (String node : strings) {
            if (skip.contains(node)) {
                if ("..".equals(node)) {
                    if (!stack.isEmpty()) {
                        stack.pop();
                    }
                }
            } else {
                stack.push(node);
            }
        }
        if (stack.isEmpty()) {
            return "/";
        }
        StringBuilder stringBuilder = new StringBuilder();
        for (String node : stack) {
            stringBuilder = stringBuilder.append("/").append(node);
        }
        return stringBuilder.toString();
    }

    /**
     * 72. Edit Distance
     *
     * @param word1
     * @param word2
     * @return
     */
    public int minDistance(String word1, String word2) {
        int m = word1.length();
        int n = word2.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (i == 0 && j == 0) {
                    dp[i][j] = 0;
                } else if (i == 0) {

                }
            }
        }
        return 0;
    }
}