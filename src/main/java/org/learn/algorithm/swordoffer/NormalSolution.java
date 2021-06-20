package org.learn.algorithm.swordoffer;

import org.learn.algorithm.datastructure.ListNode;
import org.learn.algorithm.datastructure.TreeNode;
import org.slf4j.event.SubstituteLoggingEvent;
import org.springframework.util.ResourceUtils;

import java.awt.*;
import java.util.*;
import java.util.List;

/**
 * @author dora
 * @date 2021/6/10
 */
public class NormalSolution {
    public static void main(String[] args) {
        NormalSolution solution = new NormalSolution();

        String param = "25525522135";

        solution.restoreIpAddresses(param);
    }

    public ListNode ReverseList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode reverseList = ReverseList(head.next);

        head.next.next = head;

        head.next = null;

        return reverseList;
    }

    /**
     * todo
     * lru design
     *
     * @param operators int整型二维数组 the ops
     * @param k         int整型 the k
     * @return int整型一维数组
     */
    public int[] LRU(int[][] operators, int k) {
        // write code here
        if (operators == null || operators.length == 0) {
            return new int[]{-1};
        }
        return null;
    }

    public boolean hasCycle(ListNode head) {
        if (head == null || head.next == null) {
            return false;
        }
        ListNode fast = head;
        ListNode slow = head;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (fast == slow) {
                return true;
            }
        }
        return false;
    }


    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * <p>
     * 如果目标值存在返回下标，否则返回 -1
     *
     * @param nums   int整型一维数组
     * @param target int整型
     * @return int整型
     */
    public int search(int[] nums, int target) {
        // write code here
        if (nums == null || nums.length == 0) {
            return -1;
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
        return nums[left] == target ? left : -1;
    }

    public int jumpFloor(int target) {
        if (target <= 2) {
            return target;
        }
        return jumpFloor(target - 1) + jumpFloor(target - 2);
    }


    public int maxLength(int[] arr) {
        // write code here
        if (arr == null || arr.length == 0) {
            return 0;
        }
        int left = 0;
        int result = 0;
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < arr.length; i++) {
            if (map.containsKey(arr[i])) {
                left = Math.max(left, map.get(arr[i]) + 1);
            }
            result = Math.max(result, i - left + 1);
            map.put(arr[i], i);
        }
        return result;
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

    public int maxLengthV2(int[] arr) {
        if (arr == null || arr.length == 0) {
            return 0;
        }
        int[] hash = new int[10];
        int result = 0;
        int left = 0;
        for (int i = 0; i < arr.length; i++) {
            int val = arr[i];

            left = Math.max(left, hash[val]);

            result = Math.max(result, i - left + 1);

            hash[val] = i + 1;
        }
        return result;
    }


    /**
     * @param head ListNode类
     * @param k    int整型
     * @return ListNode类
     */
    public ListNode reverseKGroupV2(ListNode head, int k) {
        // write code here
        if (head == null || head.next == null) {
            return head;
        }
        ListNode current = head;
        int count = 0;
        while (current != null && count != k) {
            current = current.next;
            count++;
        }
        if (count == k) {
            ListNode reverseNode = reverseKGroup(current, k);
            while (count-- > 0) {
                ListNode tmp = head.next;
                head.next = reverseNode;
                reverseNode = head;
                head = tmp;
            }
            head = reverseNode;
        }
        return head;

    }


    public ListNode reverseKGroup(ListNode head, int k) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode node = head;
        for (int i = 0; i < k; i++) {
            if (node == null) {
                return head;
            }
            node = node.next;
        }
        ListNode reverse = reverse(head, node);

        head.next = reverseKGroup(node, k);

        return reverse;
    }

    private ListNode reverse(ListNode start, ListNode end) {
        ListNode prev = end;
        while (start != end) {
            ListNode tmp = start.next;
            start.next = prev;
            prev = start;
            start = tmp;
        }
        return prev;
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
     * @param x int整型
     * @return int整型
     */
    public int sqrt(int x) {
        // write code here
        double precision = 0.0001;
        double result = x;
        while (result * result - x > precision) {
            result = (result + x / result) / 2;
        }
        return (int) result;
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
     * @param n    int整型
     * @return ListNode类
     */
    public ListNode removeNthFromEnd(ListNode head, int n) {
        // write code here
        if (head == null) {
            return null;
        }
        int count = 1;
        ListNode node = head;
        while (node.next != null) {
            count++;
            node = node.next;
        }
        ListNode root = new ListNode(0);

        root.next = head;

        ListNode fast = root;
        for (int i = 0; i < count - n; i++) {
            fast = fast.next;
        }
        fast.next = fast.next.next;

        return root.next;
    }

    /**
     * @param strs string字符串一维数组
     * @return string字符串
     */
    public String longestCommonPrefix(String[] strs) {
        // write code here

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
     * @param k    int整型
     * @return ListNode类
     */


    /**
     * return topK string
     *
     * @param strings string字符串一维数组 strings
     * @param k       int整型 the k
     * @return string字符串二维数组
     */
    public String[][] topKstrings(String[] strings, int k) {
        // write code here
        if (strings == null || strings.length == 0) {
            return new String[][]{};
        }
        Map<String, Integer> map = new HashMap<>();
        PriorityQueue<String> queue = new PriorityQueue<>(new Comparator<String>() {
            @Override
            public int compare(String o1, String o2) {
                Integer count1 = map.getOrDefault(o1, 0);
                Integer count2 = map.getOrDefault(o2, 0);
                if (count1.equals(count2)) {
                    return o2.compareTo(o1);
                }
                return count1.compareTo(count2);
            }
        });
        for (String item : strings) {
            Integer count = map.getOrDefault(item, 0);
            map.put(item, count + 1);
        }
        for (Map.Entry<String, Integer> item : map.entrySet()) {
            String key = item.getKey();
            queue.offer(key);
            if (queue.size() > k) {
                queue.poll();
            }
        }
        String[][] result = new String[k][2];
        int iterator = k - 1;
        while (!queue.isEmpty()) {
            String tmp = queue.poll();
            result[iterator][0] = tmp;
            result[iterator][1] = map.get(tmp) + "";
            iterator--;
        }
        return result;
    }


    /**
     * @param l1 ListNode类
     * @param l2 ListNode类
     * @return ListNode类
     */
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        // write code here
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
     * @param head ListNode类 the head node
     * @return ListNode类
     */
    public ListNode sortInList(ListNode head) {
        // write code here
        if (head == null || head.next == null) {
            return head;
        }
        ListNode fast = head;
        ListNode slow = head;
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        ListNode next = slow.next;

        slow.next = null;

        ListNode first = sortInList(head);

        ListNode second = sortInList(next);

        return mergeTwoLists(first, second);
    }


    /**
     * @param head ListNode类 the head
     * @return bool布尔型
     */
    public boolean isPail(ListNode head) {
        // write code here
        if (head == null || head.next == null) {
            return true;
        }
        ListNode fast = head;
        ListNode slow = head;
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        ListNode tmp = slow.next;
        slow.next = null;
        ListNode reverse = reverse(tmp);

        while (head != null && reverse != null) {
            if (head.val != reverse.val) {
                return false;
            }
            head = head.next;
            reverse = reverse.next;
        }
        return true;
    }


    /**
     * max increasing subsequence
     *
     * @param arr int整型一维数组 the array
     * @return int整型
     */
    public int MLS(int[] arr) {
        // write code here
        if (arr == null) {
            return 0;
        }
        int result = 0;
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : arr) {
            if (!map.containsKey(num)) {
                Integer left = map.getOrDefault(num - 1, 0);

                Integer right = map.getOrDefault(num + 1, 0);
                int val = left + right + 1;

                result = Math.max(result, val);


                map.put(num, val);
                map.put(num - left, val);
                map.put(num + right, val);
            }
        }
        return result;
    }


    /**
     * todo
     * retrun the longest increasing subsequence
     *
     * @param arr int整型一维数组 the array
     * @return int整型一维数组
     */
    public int[] LIS(int[] arr) {
        // write code here
        return null;
    }


    /**
     * 判断岛屿数量
     *
     * @param grid char字符型二维数组
     * @return int整型
     */
    public int solve(char[][] grid) {
        // write code here
        return -1;
    }

    /**
     * @param n int整型
     * @return string字符串ArrayList
     */
    public ArrayList<String> generateParenthesis(int n) {
        // write code here
        if (n <= 0) {
            return new ArrayList<>();
        }
        ArrayList<String> result = new ArrayList<>();
        intervalPath(result, 0, 0, n, "");
        return result;
    }

    private void intervalPath(ArrayList<String> result, int open, int close, int n, String s) {
        if (s.length() == 2 * n) {
            result.add(s);
            return;
        }
        if (open < n) {
            intervalPath(result, open + 1, close, n, s + "(");
        }
        if (close < open) {
            intervalPath(result, open, close + 1, n, s + ")");
        }
    }


    /**
     * @param t1 TreeNode类
     * @param t2 TreeNode类
     * @return TreeNode类
     */
    public TreeNode mergeTrees(TreeNode t1, TreeNode t2) {
        // write code here
        if (t1 == null && t2 == null) {
            return null;
        }
        if (t1 == null) {
            return t2;
        }
        if (t2 == null) {
            return t1;
        }
        TreeNode root = new TreeNode(t1.val + t2.val);
        root.left = mergeTrees(t1.left, t2.left);
        root.right = mergeTrees(t1.right, t2.right);
        return root;
    }

    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * 计算
     *
     * @param n     int整型 数组的长度
     * @param array int整型一维数组 长度为n的数组
     * @return long长整型
     */
    public long subsequence(int n, int[] array) {
        // write code here
        if (array == null || array.length == 0) {
            return 0;
        }
        int[] dp = new int[n + 1];
        dp[0] = 0;
        dp[1] = array[0];
        for (int i = 2; i <= n; i++) {
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + array[i - 1]);
        }
        return Math.max(dp[n - 1], dp[n]);
    }

    public int maxProfitII(int[] prices) {
        // write code here
        if (prices == null || prices.length == 0) {
            return 0;
        }
        int cost = prices[0];
        int result = 0;
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > cost) {
                result += prices[i] - cost;
            }
            cost = prices[i];
        }
        return result;
    }


    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * 计算成功举办活动需要多少名主持人
     *
     * @param n        int整型 有n个活动
     * @param startEnd int整型二维数组 startEnd[i][0]用于表示第i个活动的开始时间，startEnd[i][1]表示第i个活动的结束时间
     * @return int整型
     */
    public int minmumNumberOfHost(int n, int[][] startEnd) {
        // write code here
        if (startEnd == null || startEnd.length == 0) {
            return 0;
        }
        Arrays.sort(startEnd, Comparator.comparingInt(o -> o[0]));
        PriorityQueue<Integer> priorityQueue = new PriorityQueue<>();
        for (int[] item : startEnd) {
            if (!priorityQueue.isEmpty() && priorityQueue.peek() < item[0]) {
                priorityQueue.poll();
            }
            priorityQueue.offer(item[1]);
        }
        return priorityQueue.size();
    }

    public int NumberOf1(int n) {
        int count = 0;
        while (n != 0) {
            n = n & (n - 1);
            count++;
        }
        return count;
    }


    /**
     * longest common substring
     *
     * @param str1 string字符串 the string
     * @param str2 string字符串 the string
     * @return string字符串
     */
    public String LCSII(String str1, String str2) {
        // write code here
        if (str1 == null || str2 == null) {
            return "";
        }
        int m = str1.length();
        int n = str2.length();
        int[][] dp = new int[m + 1][n + 1];
        int max = 0;
        int index = 0;
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (str1.charAt(i - 1) == str2.charAt(j - 1)) {
                    dp[i][j] = 1 + dp[i - 1][j - 1];
                }
                if (dp[i][j] > max) {
                    max = dp[i][j];
                    index = i;
                }
            }
        }
        if (max == 0) {
            return "";
        }
        return str1.substring(index - max, index);
    }


    public int reverse(int x) {
        // write code here
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
     * 最大正方形
     *
     * @param matrix char字符型二维数组
     * @return int整型
     */
    public int solveMatrix(char[][] matrix) {
        // write code here
        if (matrix == null || matrix.length == 0) {
            return 0;
        }
        int row = matrix.length;
        int column = matrix[0].length;
        int[][] dp = new int[row + 1][column + 1];
        int result = 0;
        for (int i = 1; i <= row; i++) {
            for (int j = 1; j <= column; j++) {
                if (matrix[i - 1][j - 1] != '0') {
                    dp[i][j] = 1 + Math.min(Math.min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]);
                }
                if (dp[i][j] != 0) {
                    result = Math.max(result, dp[i][j] * dp[i][j]);

                }
            }
        }
        return result;
    }


    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param s string字符串 第一个整数
     * @param t string字符串 第二个整数
     * @return string字符串
     */
    public String solveMulti(String s, String t) {
        // write code here
        if (s == null || t == null) {
            return "";
        }
        int m = s.length();
        int n = t.length();
        int[] pos = new int[m + n];
        for (int i = m - 1; i >= 0; i--) {
            for (int j = n - 1; j >= 0; j--) {

                int val = Character.getNumericValue(s.charAt(i)) * Character.getNumericValue(t.charAt(j))
                        + pos[i + j + 1];

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
     * todo
     * 解码
     *
     * @param nums string字符串 数字串
     * @return int整型
     */
    public int solveDecode(String nums) {
        // write code here
        if (nums == null || nums.length() == 0) {
            return 0;
        }
        return -1;
    }


    /**
     * 进制转换
     *
     * @param M int整型 给定整数
     * @param N int整型 转换到的进制
     * @return string字符串
     */
    public String solveBit(int M, int N) {
        // write code here
        return "";
    }

    public ArrayList<ArrayList<Integer>> threeSum(int[] num) {
        if (num == null || num.length == 0) {
            return new ArrayList<>();
        }
        Arrays.sort(num);
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        for (int i = 0; i < num.length - 2; i++) {
            if (i > 0 && num[i] == num[i - 1]) {
                continue;
            }
            int left = i + 1;
            int right = num.length - 1;
            while (left < right) {
                int val = num[i] + num[left] + num[right];
                if (val == 0) {
                    ArrayList<Integer> tmp = new ArrayList<>();
                    tmp.add(num[i]);
                    tmp.add(num[left]);
                    tmp.add(num[right]);
                    while (left < right && num[left] == num[left + 1]) {
                        left++;
                    }
                    while (left < right && num[right] == num[right - 1]) {
                        right--;
                    }

                    result.add(tmp);
                    left++;
                    right--;
                } else if (val < 0) {
                    left++;
                } else {
                    right--;
                }
            }
        }
        return result;
    }

    /**
     * todo
     * 最大乘积
     *
     * @param A int整型一维数组
     * @return long长整型
     */
    public long solveThreeNum(int[] A) {
        // write code here
        if (A == null || A.length < 3) {
            return 0;
        }
        int max1 = Integer.MIN_VALUE;
        int max2 = Integer.MIN_VALUE;
        int max3 = Integer.MIN_VALUE;
        int min1 = Integer.MAX_VALUE;
        int min2 = Integer.MAX_VALUE;
        for (int num : A) {
            if (num > max1 || num > max2 || num > max3) {
                if (num > max1) {
                    max3 = max2;
                    max2 = max1;
                    max1 = num;
                } else if (num > max2) {
                    max3 = max2;
                    max2 = num;
                } else {
                    max3 = num;
                }
            }
            if (num < min1 || num < min2) {
                if (num < min1) {
                    min2 = min1;
                    min1 = num;
                } else {
                    min2 = num;
                }
            }
        }
        return Math.max((long) min1 * min2 * max1, (long) max1 * max2 * max3);
    }

    public int minNumberInRotateArray(int[] array) {
        if (array == null || array.length == 0) {
            return 0;
        }
        int left = 0;
        int right = array.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (array[mid] == array[right]) {
                right--;
            } else if (array[mid] > array[right]) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return array[left];
    }

    /**
     * @param root TreeNode类
     * @param sum  int整型
     * @return bool布尔型
     */
    public boolean hasPathSum(TreeNode root, int sum) {
        // write code here
        if (root == null) {
            return false;
        }
        return intervalHasPath(root, sum);
    }

    private boolean intervalHasPath(TreeNode root, int sum) {
        if (root == null) {
            return false;
        }
        if (root.left == null && root.right == null && sum == root.val) {
            return true;
        }
        return intervalHasPath(root.left, sum - root.val) || intervalHasPath(root.right, sum - root.val);
    }


    public void reorderList(ListNode head) {
        if (head == null || head.next == null) {
            return;
        }
        ListNode slow = head;
        ListNode fast = head;
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        ListNode next = slow.next;
        slow.next = null;
        ListNode reverse = reverse(next);

        ListNode node = head;

        while (node != null && reverse != null) {
            ListNode tmp = node.next;

            ListNode reverseNext = reverse.next;

            node.next = reverse;

            reverse.next = tmp;

            node = tmp;

            reverse = reverseNext;
        }
    }

    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param root TreeNode类
     * @return int整型一维数组
     */
    public int[] inorderTraversal(TreeNode root) {
        // write code here
        if (root == null) {
            return new int[]{};
        }
        Stack<TreeNode> stack = new Stack<>();
        TreeNode p = root;
        List<Integer> list = new ArrayList<>();
        while (!stack.isEmpty() || p != null) {
            while (p != null) {
                stack.push(p);
                p = p.left;
            }
            p = stack.pop();
            list.add(p.val);
            p = p.right;
        }
        int[] result = new int[list.size()];
        for (int i = 0; i < list.size(); i++) {
            result[i] = list.get(i);
        }
        return result;
    }


    public boolean isMatch(String s, String p) {
        if (p.isEmpty()) {
            return s.isEmpty();
        }
        int m = s.length();
        int n = p.length();
        boolean[][] dp = new boolean[m + 1][n + 1];

        dp[0][0] = true;
        for (int i = 1; i <= n; i++) {
            dp[0][i] = p.charAt(i - 1) == '*' && dp[0][i - 1];
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
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * 两次交易所能获得的最大收益
     *
     * @param prices int整型一维数组 股票每一天的价格
     * @return int整型
     */
    public int maxProfitIII(int[] prices) {
        // write code here
        if (prices == null || prices.length == 0) {
            return 0;
        }
        int[][] dp = new int[3][prices.length];

        for (int i = 1; i <= 2; i++) {
            int cost = -prices[0];
            for (int j = 1; j < prices.length; j++) {
                dp[i][j] = Math.max(dp[i][j - 1], cost + prices[j]);
                cost = Math.max(dp[i - 1][j - 1] - prices[j], cost);
            }
        }
        return dp[2][prices.length - 1];
    }

    public int maxProfitIV(int[] prices) {
        if (prices == null || prices.length == 0) {
            return 0;
        }
        int[][] dp = new int[2][prices.length];

        int cost = prices[0];
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > cost) {
                dp[0][i] = prices[i] - cost;
            } else {
                cost = prices[i];
            }
        }
        cost = -prices[0];
        for (int j = 1; j < prices.length; j++) {
            dp[1][j] = Math.max(dp[1][j - 1], cost + prices[j]);
            cost = Math.max(dp[0][j - 1] - prices[j], cost);
        }
        return Math.max(dp[0][prices.length - 1], dp[1][prices.length - 1]);
    }

    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param nums int一维数组
     * @return int二维数组
     */
    public int[][] foundMonotoneStack(int[] nums) {
        // write code here
        if (nums == null || nums.length == 0) {
            return new int[][]{};
        }
        int len = nums.length;
        int[][] result = new int[len][2];
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < nums.length; i++) {
            while (!stack.isEmpty() && nums[stack.peek()] > nums[i]) {
                stack.pop();
            }
            if (stack.isEmpty()) {
                result[i][0] = -1;
            } else {
                result[i][0] = stack.peek();
            }
            stack.push(i);
        }
        stack.clear();
        for (int i = nums.length - 1; i >= 0; i--) {
            while (!stack.isEmpty() && nums[stack.peek()] > nums[i]) {
                stack.pop();
            }
            if (stack.isEmpty()) {
                result[i][1] = -1;
            } else {
                result[i][1] = stack.peek();
            }
            stack.push(i);
        }
        return result;
    }

    /**
     * @param head ListNode类
     * @param x    int整型
     * @return ListNode类
     */
    public ListNode partition(ListNode head, int x) {
        // write code here
        if (head == null) {
            return null;
        }
        ListNode slow = new ListNode(0);
        ListNode s1 = slow;

        ListNode fast = new ListNode(0);
        ListNode f1 = fast;

        while (head != null) {
            if (head.val < x) {
                s1.next = head;
                s1 = s1.next;
            } else {
                f1.next = head;
                f1 = f1.next;
            }
            head = head.next;
        }
        f1.next = null;

        s1.next = fast.next;

        return slow.next;
    }


    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param pHead ListNode类
     * @param k     int整型
     * @return ListNode类
     */
    public ListNode FindKthToTail(ListNode pHead, int k) {
        // write code here
        if (pHead == null) {
            return null;
        }
        ListNode root = new ListNode(0);
        root.next = pHead;
        ListNode fast = root;

        for (int i = 0; i < k; i++) {
            fast = fast.next;
            if (fast == null) {
                return null;
            }
        }

        ListNode slow = root;

        while (fast.next != null) {
            slow = slow.next;
            fast = fast.next;
        }

        return slow.next;
    }

    /**
     * todo
     * 寻找最后的山峰
     *
     * @param a int整型一维数组
     * @return int整型
     */
    public int solveFindPeek(int[] a) {
        // write code here
        int index = a.length - 1;
        while (index >= 0) {
            if (index == a.length - 1) {
                if (a[index] > a[index - 1]) {
                    return index;
                }
            } else if (index == 0) {
                if (a[index] > a[index + 1]) {
                    return index;
                }
            } else {
                if (a[index] >= a[index + 1] && a[index - 1] <= a[index]) {
                    return index;
                }
            }
            index--;
        }
        return -1;
    }

    /**
     * return the min number
     *
     * @param arr int整型一维数组 the array
     * @return int整型
     */
    public int minNumberdisappered(int[] arr) {
        // write code here
        if (arr == null || arr.length == 0) {
            return -1;
        }
        for (int i = 0; i < arr.length; i++) {
            while (arr[i] > 0 && arr[i] <= arr.length && arr[i] != arr[arr[i] - 1]) {
                swap(arr, i, arr[i] - 1);
            }
        }
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] != i + 1) {
                return i + 1;
            }
        }
        return -1;
    }

    private void swap(int[] nums, int i, int j) {
        int val = nums[i];
        nums[i] = nums[j];
        nums[j] = val;
    }


    /**
     * the number of 0
     *
     * @param n long长整型 the number
     * @return long长整型
     */
    public long thenumberof0(long n) {
        // write code here

        long count = 0;

        while (n / 5 != 0) {
            count += n / 5;
            n /= 5;
        }
        return count;
    }

    /**
     * @param s string字符串
     * @return string字符串ArrayList
     */
    public ArrayList<String> restoreIpAddresses(String s) {
        // write code here
        if (s == null || s.length() < 4) {
            return new ArrayList<>();
        }
        ArrayList<String> result = new ArrayList<>();

        int len = s.length();
        for (int i = 1; i < 4 && i < len - 2; i++) {
            for (int j = i + 1; j < i + 4 && j < len - 1; j++) {
                for (int k = j + 1; k < j + 4 && k < len; k++) {
                    String a = s.substring(0, i);
                    String b = s.substring(i, j);
                    String c = s.substring(j, k);
                    String d = s.substring(k);
                    if (checkValid(a) && checkValid(b) && checkValid(c) && checkValid(d)) {
                        result.add(a + "." + b + "." + c + "." + d);
                    }
                }
            }
        }
        return result;

    }

    private boolean checkValid(String s) {
        if (s.isEmpty()) {
            return false;
        }
        int val = Integer.parseInt(s);
        if (val < 0 || val > 255) {
            return false;
        }
        int m = s.length();
        return m <= 1 || s.charAt(0) != '0';
    }

    /**
     * todo
     * max length of the subarray sum = k
     *
     * @param arr int整型一维数组 the array
     * @param k   int整型 target
     * @return int整型
     */
    public int maxlenEqualK(int[] arr, int k) {
        // write code here
        if (arr == null || arr.length == 0) {
            return 0;
        }
        Arrays.sort(arr);
        for (int i = 0; i < arr.length; i++) {
            if (i > 0 && arr[i] == arr[i - 1]) {
                continue;
            }
            int left = i + 1;
            int right = arr.length - 1;
            while (left < right) {
                int val = arr[i] + arr[left] + arr[right];
                if (val < k) {
                    left++;
                } else if (val > k) {
                    right--;
                } else {
                }
            }
        }
        return -1;
    }

    public ArrayList<ArrayList<Integer>> subsets(int[] S) {
        if (S == null || S.length == 0) {
            return new ArrayList<>();
        }
        Arrays.sort(S);
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        intervalSubSet(result, new ArrayList<Integer>(), 0, S);
        return result;
    }

    private void intervalSubSet(ArrayList<ArrayList<Integer>> result, ArrayList<Integer> integers, int start, int[] s) {
        result.add(new ArrayList<>(integers));
        for (int i = start; i < s.length; i++) {
            integers.add(s[i]);
            intervalSubSet(result, integers, i + 1, s);
            integers.remove(integers.size() - 1);
        }
    }


    /**
     * @param x int整型
     * @return bool布尔型
     */
    public boolean isPalindrome(int x) {
        // write code here
        if (x < 0) {
            return false;
        }
        if (x == 0) {
            return true;
        }
        int result = 0;
        while (x > result) {
            result = result * 10 + x % 10;
            x /= 10;
        }
        return x == result || result / 10 == x;
    }


    /**
     * @param m int整型
     * @param n int整型
     * @return int整型
     */
    public int uniquePaths(int m, int n) {
        // write code here
        int[][] dp = new int[m][n];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (i == 0 && j == 0) {
                    dp[i][j] = 1;
                } else if (i == 0) {
                    dp[i][j] = dp[i][j - 1];
                } else if (j == 0) {
                    dp[i][j] = dp[i - 1][j];
                } else {
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
                }
            }
        }
        return dp[m - 1][n - 1];
    }

    public double maxProduct(double[] arr) {
        double max = arr[0];
    }


}