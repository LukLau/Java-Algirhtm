package org.learn.algorithm.swordoffer;

import org.learn.algorithm.datastructure.ListNode;
import org.learn.algorithm.datastructure.TreeNode;

import java.util.*;

/**
 * @author dora
 * @date 2021/6/11
 */
public class NormalSolution {

    public static void main(String[] args) {
        NormalSolution solution = new NormalSolution();
        solution.solve("1", "99");
    }

    public ArrayList<Integer> GetLeastNumbers_Solution(int[] input, int k) {
        if (input == null || input.length == 0) {
            return new ArrayList<>();
        }
        if (k < 0 || k > input.length) {
            return new ArrayList<>();
        }
        PriorityQueue<Integer> priorityQueue = new PriorityQueue<>(Comparator.reverseOrder());
        for (int num : input) {
            priorityQueue.offer(num);
            if (priorityQueue.size() > k) {
                priorityQueue.poll();
            }
        }
        return new ArrayList<>(priorityQueue);
    }


    public ArrayList<Integer> GetLeastNumbers_SolutionV2(int[] input, int k) {
        if (input == null || input.length == 0 || k < 0 || k > input.length) {
            return new ArrayList<>();
        }
        k--;
        int index = getPartition(input, 0, input.length - 1);
        while (index != k) {
            if (index > k) {
                index = getPartition(input, 0, index - 1);
            } else {
                index = getPartition(input, index + 1, k);
            }
        }
        ArrayList<Integer> result = new ArrayList<>();
        for (int i = 0; i <= k; i++) {
            result.add(input[i]);
        }
        return result;
    }

    private int getPartition(int[] nums, int low, int high) {
        int pivot = nums[low];
        while (low < high) {
            while (low < high && nums[high] >= pivot) {
                high--;
            }
            if (low < high) {
                nums[low] = nums[high];
                low++;
            }
            while (low < high && nums[low] <= pivot) {
                low++;
            }
            if (low < high) {
                nums[high] = nums[low];
                high--;
            }
        }
        nums[low] = pivot;
        return low;
    }


    public void merge(int A[], int m, int B[], int n) {
        int len = m + n - 1;
        m--;
        n--;
        while (m >= 0 && n >= 0) {
            if (A[m] > B[n]) {
                A[len--] = A[m--];
            } else {
                A[len--] = B[n--];
            }
        }
        while (n >= 0) {
            A[len--] = B[n--];
        }
    }

    public int findKth(int[] a, int n, int K) {
        // write code here
        PriorityQueue<Integer> priorityQueue = new PriorityQueue<>(Comparator.naturalOrder());

        for (int num : a) {
            priorityQueue.offer(num);
            if (priorityQueue.size() > K) {
                priorityQueue.poll();
            }
        }
        return priorityQueue.peek();
    }


    public String solve(String str) {
        // write code here
        if (str == null || str.isEmpty()) {
            return "";
        }
        StringBuilder builder = new StringBuilder();
        return builder.append(str).reverse().toString();
    }


    public int getLongestPalindrome(String A, int n) {
        // write code here
        if (A == null || A.isEmpty()) {
            return 0;
        }
        int m = A.length();
        boolean[][] dp = new boolean[m][m];
        int result = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j <= i; j++) {
                if (A.charAt(j) == A.charAt(i) && (i - j <= 1 || dp[j + 1][i - 1])) {
                    dp[j][i] = true;
                }

                if (dp[j][i] && i - j + 1 > result) {
                    result = i - j + 1;
                }
            }
        }
        return result;
    }


    public int calculate(String s) {
        // write code here
        if (s == null) {
            return 0;
        }
        s = s.trim();
        if (s.isEmpty()) {
            return 0;
        }
        Stack<Integer> stack = new Stack<>();
        int sign = 1;
        char[] words = s.toCharArray();
        int endIndex = 0;
        int result = 0;
        while (endIndex < words.length) {
            if (Character.isDigit(words[endIndex])) {
                int tmp = 0;
                while (endIndex < words.length && Character.isDigit(words[endIndex])) {
                    tmp = tmp * 10 + Character.getNumericValue(words[endIndex]);
                    endIndex++;
                }
                result += tmp * sign;
            }
            if (endIndex == words.length || words[endIndex] != ' ') {

            }

        }
        return -1;
    }


    /**
     * @param root TreeNode类
     * @param sum  int整型
     * @return int整型ArrayList<ArrayList <>>
     */
    public ArrayList<ArrayList<Integer>> pathSum(TreeNode root, int sum) {
        // write code here
        if (root == null) {
            return new ArrayList<>();
        }
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        intervalPathSum(result, new ArrayList<>(), root, sum);
        return result;
    }

    private void intervalPathSum(ArrayList<ArrayList<Integer>> result, ArrayList<Integer> tmp, TreeNode root, int sum) {
        tmp.add(root.val);
        if (root.left == null && root.right == null && root.val == sum) {
            result.add(new ArrayList<>(tmp));
        } else {
            if (root.left != null) {
                intervalPathSum(result, tmp, root.left, sum - root.val);
            }
            if (root.right != null) {
                intervalPathSum(result, tmp, root.right, sum - root.val);
            }
        }
        tmp.remove(tmp.size() - 1);
    }


    /**
     * @param head1 ListNode类
     * @param head2 ListNode类
     * @return ListNode类
     */
    public ListNode addInList(ListNode head1, ListNode head2) {
        // write code here
        if (head1 == null && head2 == null) {
            return null;
        }
        head1 = reverse(head1);

        head2 = reverse(head2);

        if (head1 == null || head2 == null) {
            return head1 == null ? head2 : head1;
        }
        ListNode root = new ListNode(0);
        ListNode dummy = root;
        int carry = 0;
        while (head1 != null || head2 != null || carry != 0) {
            int val = (head1 == null ? 0 : head1.val) + (head2 == null ? 0 : head2.val) + carry;

            dummy.next = new ListNode(val % 10);

            dummy = dummy.next;

            carry = val / 10;

            head1 = head1 == null ? null : head1.next;

            head2 = head2 == null ? null : head2.next;
        }
        return root.next;
    }

    private ListNode reverse(ListNode head) {
        if (head == null) {
            return null;
        }
        ListNode prev = null;
        while (head != null) {
            ListNode tmp = head.next;
            head.next = prev;
            prev = head;
            head = tmp;
        }
        return prev;
    }


    /**
     * @param prices int整型一维数组
     * @return int整型
     */
    public int maxProfit(int[] prices) {
        // write code here
        if (prices == null || prices.length == 0) {
            return 0;
        }
        int cost = prices[0];
        int result = 0;
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > cost) {
                result = Math.max(result, prices[i] - cost);
            } else {
                cost = prices[i];
            }
        }
        return result;
    }


    public String solve(String s, String t) {
        // write code here
        if (s == null || t == null) {
            return "";
        }
        int m = s.length() - 1;
        int n = t.length() - 1;
        int carry = 0;
        StringBuilder builder = new StringBuilder();
        while (m >= 0 || n >= 0 || carry != 0) {
            int val = (m >= 0 ? Character.getNumericValue(s.charAt(m)) : 0)
                    + (n >= 0 ? Character.getNumericValue(t.charAt(n)) : 0) + carry;

            carry = val / 10;

            builder.append(val % 10);

            m--;
            n--;
        }

        return builder.reverse().toString();
    }


    public int foundOnceNumber(int[] arr, int k) {
        // write code here
//        int[] count = new int[32];
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : arr) {
            Integer count = map.getOrDefault(num, 0);
            map.put(num, count + 1);
        }
        for (Map.Entry<Integer, Integer> item : map.entrySet()) {
            Integer value = item.getValue();

            if (value % k != 0) {
                return item.getKey();
            }
        }
        return -1;
    }


    /**
     * todo
     *
     * @param arr
     * @param k
     * @return
     */
    public int foundOnceNumberV2(int[] arr, int k) {
        int[] count = new int[32];
        for (int i = 0; i < 32; i++) {
            for (int num : arr) {
                if ((num & 1 << i) != 0) {
                    count[i]++;
                }
            }
        }
        int result = 0;
        for (int i = 0; i < 32; i++) {
            if (count[i] % k != 0) {
                result += 1 << i;
            }
        }
        return result;
    }


    public long maxWater(int[] arr) {
        // write code here
        if (arr == null || arr.length == 0) {
            return 0;
        }
        long result = 0;
        int leftEdge = 0;
        int rightEdge = 0;
        int left = 0;
        int right = arr.length - 1;
        while (left < right) {
            if (arr[left] < arr[right]) {
                if (arr[left] > leftEdge) {
                    leftEdge = arr[left];
                } else {
                    result += leftEdge - arr[left];
                }
                left++;
            } else {
                if (arr[right] > rightEdge) {
                    rightEdge = arr[right];
                } else {
                    result += rightEdge - arr[right];
                }
                right--;
            }
        }
        return result;
    }


    /**
     * longest common subsequence
     *
     * @param s1 string字符串 the string
     * @param s2 string字符串 the string
     * @return string字符串
     */
    public String LCS(String s1, String s2) {
        // write code here
        if (s1 == null || s2 == null) {
            return "";
        }
        int m = s1.length();
        int n = s2.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (s1.charAt(i - 1) == s2.charAt(j - 1)) {
                    dp[i][j] = 1 + dp[i - 1][j - 1];
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        StringBuilder builder = new StringBuilder();
        int i = m;
        int j = n;
        while (i > 0 && j > 0) {
            if (s1.charAt(i - 1) == s2.charAt(j - 1)) {
                builder.append(s1.charAt(i - 1));
                i--;
                j--;
            } else {
                if (dp[i - 1][j] < dp[i][j - 1]) {
                    j--;
                } else {
                    i--;
                }
            }
        }
        return builder.length() == 0 ? "-1" : builder.reverse().toString();
    }


    /**
     * @param head ListNode类
     * @param k    int整型
     * @return ListNode类
     */
    public ListNode reverseKGroup(ListNode head, int k) {
        // write code here
        return null;
    }


}
