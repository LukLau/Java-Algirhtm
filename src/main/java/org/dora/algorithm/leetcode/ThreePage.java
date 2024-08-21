package org.dora.algorithm.leetcode;

import org.dora.algorithm.datastructe.ListNode;
import org.dora.algorithm.datastructe.TreeNode;

import java.util.*;

/**
 * @author dora
 * @date 2019-05-02
 */
public class ThreePage {


    public static void main(String[] args) {
        ThreePage threePage = new ThreePage();

//        String calculate = "1+(1-1)";
//        threePage.calculate(calculate);
        String calculateII = " 3+5 / 2 ";
//        threePage.calculateII(calculateII);
        int[] slideWindows = new int[]{1, 3, -1, -3, 5, 3, 6, 7};

        int[] result = threePage.maxSlidingWindow(slideWindows, 3);
        Arrays.stream(result).peek(System.out::println);
    }

    /**
     * 201. Bitwise AND of Numbers Range
     * todo 不懂 位运算
     *
     * @param m
     * @param n
     * @return
     */
    public int rangeBitwiseAnd(int m, int n) {
        return 0;
    }

    /**
     * 202. Happy Number
     *
     * @param n
     * @return
     */
    public boolean isHappy(int n) {
        if (n <= 0) {
            return false;
        }
        Set<Integer> used = new HashSet<>();
        while (n != 0) {
            int tmp = n;
            int result = 0;
            while (tmp != 0) {
                int value = tmp % 10;
                result += value * value;

                tmp /= 10;
            }

            if (result == 1) {
                return true;
            }

            if (used.contains(result)) {
                return false;
            }
            n = result;
            used.add(n);
        }
        return false;
    }

    /**
     * 203. Remove Linked List Elements
     *
     * @param head
     * @param val
     * @return
     */
    public ListNode removeElements(ListNode head, int val) {
        if (head == null) {
            return null;
        }

        if (head.val == val) {
            return this.removeElements(head.next, val);
        } else {
            head.next = this.removeElements(head.next, val);
            return head;
        }
    }

    /**
     * 204. Count Primes 计算素数个数
     * todo 巧妙设计
     *
     * @param n
     * @return
     */
    public int countPrimes(int n) {
        int count = 0;
        for (int i = 2; i < Math.sqrt(n); i++) {
            if (this.isPrime(i)) {
                count++;
            }
        }
        return count;
    }

    private boolean isPrime(int i) {
        for (int j = 2; j < i; j++) {
            if (i % j == 0) {
                return false;
            }
        }
        return true;
    }

    /**
     * 205. Isomorphic Strings
     * todo 哈希思想 注意遍历退出条件
     *
     * @param s
     * @param t
     * @return
     */
    public boolean isIsomorphic(String s, String t) {
        if (s == null || t == null) {
            return false;
        }
        int[] hash1 = new int[512];
        int[] hash2 = new int[512];
        for (int i = 0; i < s.length(); i++) {
            if (hash1[s.charAt(i)] != hash2[t.charAt(i)]) {
                return false;
            }
            hash1[s.charAt(i)] = i + 1;
            hash2[t.charAt(i)] = i + 1;
        }
        return false;
    }

    /**
     * 反转链表
     *
     * @param head
     * @return
     */
    public ListNode reverseList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode next = this.reverseList(head.next);
        head.next.next = head;
        head.next = null;
        return next;

    }

    /**
     * 209. Minimum Size Subarray Sum
     *
     * @param s
     * @param nums
     * @return
     */
    public int minSubArrayLen(int s, int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int begin = 0;
        int end = 0;
        int result = Integer.MAX_VALUE;
        int local = 0;
        while (end < nums.length) {

            local += nums[end];


            while (local >= s) {

                result = Math.min(result, end - begin + 1);

                local -= nums[begin++];
            }
            end++;
        }
        return result == Integer.MAX_VALUE ? 0 : result;
    }

    /**
     * 213. House Robber II
     *
     * @param nums
     * @return
     */
    public int rob(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        return Math.max(this.houseRob(0, nums.length - 2, nums), this.houseRob(1, nums.length - 1, nums));
    }

    private int houseRob(int start, int end, int[] nums) {
        if (start > end) {
            return 0;
        }
        int robPrev = 0;
        int robCurrent = 0;
        for (int i = start; i <= end; i++) {
            int tmp = robPrev;
            robPrev = Math.max(robPrev, robCurrent);
            robCurrent = tmp + nums[i];
        }
        return Math.max(robPrev, robCurrent);
    }

    /**
     * 214. Shortest Palindrome
     *
     * @param s
     * @return
     */
    public String shortestPalindrome(String s) {
        if (s == null || s.length() == 0) {
            return "";
        }
        return "";
    }

    /**
     * 214. Shortest Palindrome
     *
     * @param nums
     * @param k
     * @return
     */
    public int findKthLargest(int[] nums, int k) {
        if (nums == null || nums.length == 0) {
            return -1;
        }

        int partition = this.partition(nums, 0, nums.length - 1);
        k--;
        while (partition != k) {

            if (partition > k) {
                partition = this.partition(nums, 0, k - 1);
            } else {
                partition = this.partition(nums, partition + 1, nums.length - 1);
            }
        }
        return nums[k];
    }

    private int partition(int[] nums, int start, int end) {
        if (start > end) {
            return -1;
        }
        int pivot = nums[start];
        while (start < end) {
            while (start < end && nums[end] >= pivot) {
                end--;
            }
            if (start < end) {
                nums[start] = nums[end];
                start++;
            }
            while (start < end && nums[start] <= pivot) {
                start++;
            }
            if (start < end) {
                nums[end] = nums[start];
                end--;
            }
        }
        nums[start] = pivot;
        return start;
    }

    /**
     * 216. Combination Sum III
     *
     * @param k
     * @param n
     * @return
     */
    public List<List<Integer>> combinationSum3(int k, int n) {
        if (k <= 0 || n <= 0) {
            return new ArrayList<>();
        }
        List<List<Integer>> result = new ArrayList<>();
        internalCombinationSum3(result, new ArrayList<>(), 1, k, n);
        return result;
    }

    private void internalCombinationSum3(List<List<Integer>> result, List<Integer> tmp, int index, int k, int n) {
        if (tmp.size() > k || n < 0) {
            return;
        }
        if (tmp.size() == k && n == 0) {
            result.add(new ArrayList<>(tmp));
            return;
        }
        for (int i = index; i <= 9 && i <= n; i++) {
            tmp.add(i);
            internalCombinationSum3(result, tmp, i + 1, k, n - i);
            tmp.remove(tmp.size() - 1);
        }
    }


    /**
     * 219. Contains Duplicate II
     *
     * @param nums
     * @param k
     * @return
     */
    public boolean containsNearbyDuplicate(int[] nums, int k) {
        if (nums == null || nums.length == 0) {
            return false;
        }
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(nums[i])) {
                int diff = Math.abs(i - map.get(nums[i]));
                if (diff > k) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * 220. Contains Duplicate III
     *
     * @param nums
     * @param k
     * @param t
     * @return
     */
    public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
        if (nums == null || nums.length == 0) {
            return false;
        }
        TreeSet<Integer> treeSet = new TreeSet<>();
        for (int i = 0; i < nums.length; i++) {
            Integer floor = treeSet.floor(i - t);
            Integer ceil = treeSet.ceiling(i + t);
        }
        return false;

    }

    /**
     * 221. Maximal Square
     * todo 需考虑好方程式
     *
     * @param matrix
     * @return
     */
    public int maximalSquare(char[][] matrix) {
        if (matrix == null | matrix.length == 0) {
            return 0;
        }
        int row = matrix.length;
        int column = matrix[0].length;


        int result = 0;


        int[][] dp = new int[row + 1][column + 1];

        for (int i = 1; i <= row; i++) {
            for (int j = 1; j <= column; j++) {
                if (matrix[i - 1][j - 1] == '1') {
                    dp[i][j] = Math.min(Math.min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]) + 1;
//                    System.out.println("i:" + i + "j:" + j + "width:" + width);
                    result = Math.max(result, dp[i][j]);
                }
            }
        }
        return result * result;
    }


    /**
     * 224. Basic Calculator
     *
     * @param s
     * @return
     */
    public int calculate(String s) {
        if (s == null) {
            return 0;
        }
        s = s.trim();
        if (s.isEmpty()) {
            return 0;
        }
        int startIndex = 0;
        int len = s.length();
        char[] words = s.toCharArray();
        int result = 0;
        int sign = 1;
        while (startIndex < len) {
            if (words[startIndex] == '(') {
                int endIndex = startIndex;
                int count = 0;
                while (endIndex < words.length) {
                    if (Character.isDigit(words[endIndex])) {
                        endIndex++;
                        continue;
                    }
                    if (words[endIndex] == '(') {
                        count++;
                    }
                    if (words[endIndex] == ')') {
                        count--;
                    }
                    if (count == 0) {
                        break;
                    }
                    endIndex++;
                }
                result += sign * calculate(s.substring(startIndex + 1, endIndex));
                startIndex = endIndex + 1;
            }
            if (startIndex < words.length) {
                if (Character.isDigit(words[startIndex])) {
                    int tmp = 0;
                    while (startIndex < words.length && Character.isDigit(words[startIndex])) {
                        tmp = tmp * 10 + Character.getNumericValue(words[startIndex++]);
                    }
                    result += sign * tmp;
                } else if (words[startIndex] == '+' || words[startIndex] == '-') {
                    sign = words[startIndex] == '+' ? 1 : -1;
                    startIndex++;
                } else {
                    startIndex++;
                }
            } else {
                break;
            }
        }
        return result;
    }


    /**
     * 226. Invert Binary Tree
     *
     * @param root
     * @return
     */
    public TreeNode invertTree(TreeNode root) {
        if (root != null) {
            TreeNode left = root.left;
            root.left = root.right;
            root.right = left;
            this.invertTree(root.left);
            this.invertTree(root.right);
        }
        return root;
    }


    /**
     * 227. Basic Calculator II
     * todo 不太懂 不熟练
     *
     * @param s
     * @return
     */
    public int calculateII(String s) {
        if (s == null) {
            return 0;
        }
        s = s.trim();
        if (s.isEmpty()) {
            return 0;
        }
        Stack<Integer> stack = new Stack<>();
        char[] words = s.toCharArray();
        int index = 0;
        int sign = 1;
        char prefix = '+';
        if (words[index] == '+' || words[index] == '-') {
            prefix = words[index];
            sign = words[index++] == '+' ? 1 : -1;
        }
        while (index < words.length) {
            if (words[index] == ' ') {
                index++;
                continue;
            }
            if (Character.isDigit(words[index])) {
                int tmp = 0;
                while (index < words.length && Character.isDigit(words[index])) {
                    tmp = tmp * 10 + Character.getNumericValue(words[index++]);
                }
                stack.push(tmp);
            }
            if (prefix == '-' || prefix == '+') {
                sign = prefix == '-' ? -1 : 1;
                stack.push(sign * stack.pop());
            } else if (prefix == '*') {
                Integer secondItem = stack.pop();
                Integer firstItem = stack.pop();
                stack.push(firstItem * secondItem);
            } else if (prefix == '/') {
                Integer secondItem = stack.pop();
                Integer firstItem = stack.pop();
                stack.push(firstItem / secondItem);
            }
            if (index < words.length && !Character.isDigit(words[index])) {
                prefix = words[index++];
            }
        }
        int answer = 0;
        for (Integer number : stack) {
            answer += number;
        }
        return answer;

    }


    /**
     * 228. Summary Ranges
     * todo 巧妙设计
     *
     * @param nums
     * @return
     */
    public List<String> summaryRanges(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new ArrayList<>();
        }
        int lower = nums[0];
        List<String> result = new ArrayList<>();
        for (int i = 0; i < nums.length; i++) {
            if (i > 0 && nums[i] != nums[i - 1] + 1) {
                String range = getRange(lower, nums[i - 1]);
                result.add(range);
                lower = nums[i];
            }
        }
        String range = getRange(lower, nums[nums.length - 1]);
        if (range != null) {
            result.add(range);
        }
        return result;
    }

    private String getRange(int start, int end) {
        if (start > end) {
            return null;
        }
        return start == end ? String.valueOf(start) : start + "->" + end;
    }


    /**
     * 229. Majority Element II
     *
     * @param nums
     * @return
     */
    public List<Integer> majorityElement(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new ArrayList<>();
        }
        List<Integer> ans = new ArrayList<>();
        int candidateA = nums[0];
        int candidateB = nums[0];
        int countA = 0;
        int countB = 0;
        for (int num : nums) {
            if (num == candidateA) {
                countA++;
                continue;
            }
            if (num == candidateB) {
                countB++;
                continue;
            }
            if (countA == 0) {
                candidateA = num;
                countA = 1;
                continue;
            }
            if (countB == 0) {
                candidateB = num;
                countB = 1;
                continue;
            }
            countA--;
            countB--;
        }
        countA = 0;
        countB = 0;
        for (int num : nums) {
            if (num == candidateA) {
                countA++;
            } else if (num == candidateB) {
                countB++;
            }
        }
        if (countA * 3 > nums.length) {
            ans.add(candidateA);
        }
        if (countB * 3 > nums.length) {
            ans.add(candidateB);
        }
        return ans;
    }


    public int countDigitOne(int n) {
        int count = 0;
        for (int i = 1; i <= n; i++) {
            String numberValue = String.valueOf(i);
            char[] words = numberValue.toCharArray();
            for (char word : words) {
                if (word == '1') {
                    count++;
                }
            }
        }
        return count;
    }


    /**
     * 230. Kth Smallest Element in a BST
     *
     * @param root
     * @param k
     * @return
     */
    public int kthSmallest(TreeNode root, int k) {
        if (root == null || k <= 0) {
            return -1;
        }
        Stack<TreeNode> stack = new Stack<>();
        int count = 0;
        while (!stack.isEmpty() | root != null) {
            while (root != null) {
                stack.push(root);
                root = root.left;
            }
            root = stack.pop();
            count++;
            if (count == k) {
                return root.val;
            }
            root = root.right;
        }
        return -1;
    }

    /**
     * 231. Power of Two
     *
     * @param n
     * @return
     */
    public boolean isPowerOfTwo(int n) {
        if (n <= 0) {
            return false;
        }

        n = n & (n - 1);
        return n == 0;
    }


    /**
     * 234. Palindrome Linked List
     *
     * @param head
     * @return
     */
    public boolean isPalindrome(ListNode head) {
        if (head == null) {
            return false;
        }
        if (head.next == null) {
            return true;
        }
        ListNode fast = head;
        ListNode slow = head;
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        slow.next = this.reverseList(slow.next);
        slow = slow.next;
        while (slow != null) {
            if (head.val != slow.val) {
                return false;
            }
            head = head.next;
            slow = slow.next;
        }
        return true;

    }


    /**
     * 235. Lowest Common Ancestor of a Binary Search Tree
     *
     * @param root
     * @param p
     * @param q
     * @return
     */
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == p || root == q) {
            return root;
        }
        if ((p.val < root.val && (root.val < q.val)) || (q.val < root.val && root.val < p.val)) {
            return root;
        } else if ((p.val < root.val && q.val < root.val)) {
            return lowestCommonAncestorTree(root.left, p, q);
        } else {
            return lowestCommonAncestorTree(root.right, p, q);

        }
    }


    /**
     * 236. Lowest Common Ancestor of a Binary Tree
     *
     * @param root
     * @param p
     * @param q
     * @return
     */
    public TreeNode lowestCommonAncestorTree(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || p == root || q == root) {
            return root;
        }

        TreeNode left = this.lowestCommonAncestorTree(root.left, p, q);

        TreeNode right = this.lowestCommonAncestorTree(root.right, p, q);

        if (left == null && right == null) {
            return root;
        } else {
            return left != null ? left : right;
        }
    }


    /**
     * 237. Delete Node in a Linked List
     *
     * @param node
     */
    public void deleteNode(ListNode node) {
        if (node == null) {
            return;
        }
        if (node.next == null) {
            node = null;
        } else {
            node.val = node.next.val;

            ListNode tmp = node;

            node.next = tmp.next;

            tmp = null;

        }

    }


    /**
     * 238. Product of Array Except Self
     *
     * @param nums
     * @return
     */
    public int[] productExceptSelf(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new int[]{};
        }
        int[] ans = new int[nums.length];
        int base = 1;
        for (int i = 0; i < nums.length; i++) {
            ans[i] = base;
            base ^= nums[i];
        }
        base = 1;
        for (int i = nums.length - 1; i >= 0; i--) {
            ans[i] *= base;
            base *= ans[i];
        }
        return ans;
    }


    /**
     * 239. Sliding Window Maximum
     *
     * @param nums
     * @param k
     * @return
     */
    public int[] maxSlidingWindow(int[] nums, int k) {
        if (nums == null || nums.length == 0) {
            return new int[]{};
        }
        LinkedList<Integer> linkedList = new LinkedList<>();
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < nums.length; i++) {
            int index = i - k + 1;
            while (!linkedList.isEmpty() && nums[linkedList.peekLast()] <= nums[i]) {
                linkedList.pollLast();
            }
            linkedList.add(i);
            while (index >= 0 && linkedList.peekFirst() < index) {
                linkedList.pollFirst();
            }
            if (index >= 0) {
                result.add(nums[linkedList.peekFirst()]);
            }
        }
        int[] answer = new int[result.size()];
        for (int i = 0; i < result.size(); i++) {
            answer[i] = result.get(i);
        }
        return answer;
    }

    /**
     * 240. Search a 2D Matrix II
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

        int i = 0;
        int j = column - 1;
        while (i < row && j >= 0) {
            if (matrix[i][j] == target) {
                return true;
            } else if (matrix[i][j] > target) {
                j--;
            } else {
                i++;
            }
        }
        return false;
    }

    public List<Integer> diffWaysToCompute(String input) {
        return null;
    }


}