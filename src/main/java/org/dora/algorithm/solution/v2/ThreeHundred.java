package org.dora.algorithm.solution.v2;

import org.dora.algorithm.datastructe.ListNode;

import java.util.*;

/**
 * @author dora
 * @date 2019/9/30
 */
public class ThreeHundred {


    /**
     * todo 难题
     * 201. Bitwise AND of Numbers Range
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
        if (n < 0) {
            return false;
        }

        Set<Integer> set = new HashSet<>();

        while (n != 0) {
            int result = n;

            int value = 0;

            while (result != 0) {
                int tmp = result % 10;

                value += tmp * tmp;

                result /= 10;
            }
            if (set.contains(value)) {
                return false;
            }

            if (value == 1) {
                return true;
            }
            set.add(value);

            n = value;

        }
        return false;
    }


    /**
     * tdo 搞不懂答案
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
     * todo 求解质数
     * 204. Count Primes
     *
     * @param n
     * @return
     */
    public int countPrimes(int n) {
        if (n <= 0) {
            return 0;
        }
        int count = 0;

        for (int i = 2; i <= n; i++) {

            for (int j = 2; j * j <= n; j++) {

                if (i % 2 == 0) {
                    count++;
                }
            }
        }
        return count;
    }

    /**
     * todo 不懂
     * 205. Isomorphic Strings
     *
     * @param s
     * @param t
     * @return
     */
    public boolean isIsomorphic(String s, String t) {
        if (s == null || t == null) {
            return false;
        }
        if (s.length() != t.length()) {
            return false;
        }
        int[] hash1 = new int[256];
        int[] hash2 = new int[256];
        for (int i = 0; i < s.length(); i++) {
            hash1[s.charAt(i) - 'a']++;
            hash2[t.charAt(i) - 'a']++;
        }
        for (int i = 0; i < hash1.length; i++) {
            if (hash1[i] != hash2[i]) {
                return false;
            }
        }
        return true;
    }

    /**
     * 206. Reverse Linked List
     *
     * @param head
     * @return
     */
    public ListNode reverseList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode tmp = head.next;

        ListNode node = this.reverseList(tmp);

        tmp.next = head;

        head.next = null;

        return node;

    }

    /**
     * todo 不懂
     * 207. Course Schedule
     *
     * @param numCourses
     * @param prerequisites
     * @return
     */
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        return false;
    }

    /**
     * todo 可以考虑转化 ologn
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
        int result = Integer.MAX_VALUE;

        int begin = 0;
        int end = 0;
        int local = 0;
        while (end < nums.length) {
            local += nums[end++];

            while (local >= s) {
                result = Math.min(result, end - begin);

                local -= nums[begin++];
            }
        }
        return result == Integer.MAX_VALUE ? 0 : result;

    }

    /**
     * 210. Course Schedule II
     *
     * @param numCourses
     * @param prerequisites
     * @return
     */
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        return null;
    }

    /**
     * todo 字典记载
     * 212. Word Search II
     *
     * @param board
     * @param words
     * @return
     */
    public List<String> findWords(char[][] board, String[] words) {
        if (board == null || board.length == 0) {
            return Collections.emptyList();
        }
        Trie trie = new Trie();
        for (String word : words) {
            trie.insert(word);
        }
        List<String> ans = new ArrayList<>();

        for (int i = 0; i < board.length; i++) {

            for (int j = 0; j < board[i].length; j++) {

                if (trie.startsWith(String.valueOf(board[i][j]))) {

                }
            }
        }
        return null;
    }


    /**
     * todo 存在bug
     * 213. House Robber II
     *
     * @param nums
     * @return
     */
    public int rob(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        if (nums.length <= 2) {
            return Math.max(nums[0], nums[nums.length - 1]);
        }
        return Math.max(this.intervalRob(nums, 0, nums.length - 2),
                this.intervalRob(nums, 1, nums.length - 1));
    }

    private int intervalRob(int[] nums, int i, int j) {
        int[] dp = new int[nums.length];

        dp[0] = nums[0];

        for (int k = i; k <= j; k++) {
            if (k == 1) {
                dp[k] = Math.max(0, nums[1]);

            } else if (k > 1) {

                dp[k] = Math.max(dp[k - 2] + nums[k], dp[k - 1]);
            }
        }
        return dp[j];
    }


    /**
     * todo 不懂
     * 214. Shortest Palindrome
     *
     * @param s
     * @return
     */
    public String shortestPalindrome(String s) {
        if (s == null || s.isEmpty()) {
            return "";
        }
        return null;
    }


    /**
     * 215. Kth Largest Element in an Array
     *
     * @param nums
     * @param k
     * @return
     */
    public int findKthLargest(int[] nums, int k) {
        if (nums == null || nums.length == 0) {
            return -1;
        }
        k = nums.length - k;
        k--;
        int index = this.partition(nums, 0, nums.length - 1);
        while (index != k) {
            if (index > k) {

            }
        }
        return -1;
    }


    private int partition(int[] nums, int start, int end) {
        int pivot = nums[start];
        while (start < end) {
            while (start < end && nums[end] >= pivot) {
                end--;
            }
            if (start < end) {
                nums[end] = nums[start];
                start++;
            }
            while (start < end && nums[start] >= pivot) {
                start++;
            }
            if (start < end) {
                nums[start] = nums[end];
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
            return Collections.emptyList();
        }
        List<List<Integer>> ans = new ArrayList<>();
        this.combinationSum3(ans, new ArrayList<>(), k, 1, n);
        return ans;
    }

    private <E> void combinationSum3(List<List<Integer>> ans, List<Integer> tmp, int k, int start, int n) {
        if (tmp.size() == k && n == 0) {
            ans.add(new ArrayList<>(tmp));
            return;
        }
        for (int i = start; i <= 9 && i <= n; i++) {
            tmp.add(i);

            this.combinationSum3(ans, tmp, k, i + 1, n - i);

            tmp.remove(tmp.size() - 1);
        }

    }


}
