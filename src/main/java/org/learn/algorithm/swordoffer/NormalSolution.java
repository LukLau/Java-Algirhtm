package org.learn.algorithm.swordoffer;

import org.learn.algorithm.datastructure.Interval;
import org.learn.algorithm.datastructure.ListNode;
import org.learn.algorithm.datastructure.TreeNode;

import java.util.*;

/**
 * @author dora
 * @date 2021/6/10
 */
public class NormalSolution {
    public static void main(String[] args) {
        NormalSolution solution = new NormalSolution();

        solution.minWindow("XDOYEZODEYXNZ", "XYZ");
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

        int[] dp = new int[n];

        Arrays.fill(dp, 1);

        for (int i = 1; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (j != 0) {
                    dp[j] += dp[j - 1];
                }
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
        for (int j = 0; j < column; j++) {
            dp[j] = obstacleGrid[0][j] == 1 ? 0 : (j == 0 ? 1 : dp[j - 1]);
        }
        for (int i = 1; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (obstacleGrid[i][j] == 1) {
                    dp[j] = 0;
                } else {
                    dp[j] = dp[j] + (j == 0 ? 0 : dp[j - 1]);
                }
            }
        }
        return dp[column - 1];
    }

    public double maxProduct(double[] arr) {
        double max = arr[0];
        double min = arr[0];
        double result = arr[0];

        for (int i = 1; i < arr.length; i++) {
            double val = arr[i];
            double tmpMax = Math.max(Math.max(max * val, val * min), val);
            double tmpMin = Math.min(Math.min(max * val, val * min), val);
            result = Math.max(result, tmpMax);
            max = tmpMax;
            min = tmpMin;
        }
        return result;
    }


    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param a string字符串 待计算字符串
     * @return int整型
     */
    public int solveNotRepeatSubString(String a) {
        // write code here
        if (a == null || a.isEmpty()) {
            return 0;
        }
        int m = a.length();
        int result = 0;
        char[] words = a.toCharArray();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j <= i; j++) {
                if (checkValid(words, j, i) && i - j + 1 > result) {
                    result = i - j + 1;
                }
            }
        }
        return result;
    }

    private boolean checkValid(char[] words, int start, int end) {
        if (start == end) {
            return false;
        }
        Map<Character, Integer> map = new HashMap<>();
        for (int i = start; i <= end; i++) {
            Integer count = map.getOrDefault(words[i], 0);
            map.put(words[i], count + 1);
        }
        for (Map.Entry<Character, Integer> item : map.entrySet()) {
            Integer count = item.getValue();
            if (count % 2 != 0) {
                return false;
            }
        }
        return true;
    }

    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * <p>
     * 比较版本号
     *
     * @param version1 string字符串
     * @param version2 string字符串
     * @return int整型
     */
    public int compareVersion(String version1, String version2) {
        // write code here
        String[] word1 = version1.split("\\.");
        String[] word2 = version2.split("\\.");
        int index1 = 0;
        int index2 = 0;
        while (index1 < word1.length || index2 < word2.length) {
            int v1 = index1 == word1.length ? 0 : Integer.parseInt(word1[index1++]);
            int v2 = index2 == word2.length ? 0 : Integer.parseInt(word2[index2++]);
            if (v1 != v2) {
                return v1 - v2 < 0 ? -1 : 1;
            }
        }
        return 1;
    }


    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param str string字符串 待判断的字符串
     * @return bool布尔型
     */
    public boolean judge(String str) {
        // write code here
        if (str == null || str.isEmpty()) {
            return false;
        }
        char[] words = str.toCharArray();
        int start = 0;
        int end = words.length - 1;
        while (start < end) {
            if (words[start] != words[end]) {
                return false;
            }
            start++;
            end--;
        }
        return true;
    }

    public int MoreThanHalfNum_Solution(int[] array) {
        if (array == null || array.length == 0) {
            return 0;
        }
        int candidate = array[0];
        int count = 0;
        for (int num : array) {
            if (candidate == num) {
                count++;
            } else {
                count--;
            }
            if (count == 0) {
                candidate = num;
                count = 1;
            }
        }
        count = 0;
        for (int num : array) {
            if (num == candidate) {
                count++;
            }
        }
        return 2 * count > array.length ? candidate : -1;
    }

    public ArrayList<ArrayList<Integer>> combinationSum2(int[] num, int target) {
        if (num == null || num.length == 0) {
            return new ArrayList<>();
        }
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        Arrays.sort(num);
        intervalCombine(result, new ArrayList<Integer>(), 0, num, target);
        return result;
    }

    private void intervalCombine(ArrayList<ArrayList<Integer>> result, ArrayList<Integer> integers, int start, int[] num, int target) {
        if (target == 0) {
            result.add(new ArrayList<>(integers));
            return;
        }
        for (int i = start; i < num.length && target >= num[i]; i++) {
            if (i > start && num[i] == num[i - 1]) {
                continue;
            }
            integers.add(num[i]);
            intervalCombine(result, integers, i + 1, num, target - num[i]);
            integers.remove(integers.size() - 1);
        }
    }


    /**
     * 验证IP地址
     *
     * @param IP string字符串 一个IP地址字符串
     * @return string字符串
     */
    public String solveipv6(String IP) {
        // write code here
        if (IP == null || IP.isEmpty()) {
            return "Neither";
        }
        if (IP.contains(".")) {
            String[] words = IP.split("\\.");
            if (words.length != 4) {
                return "Neither";
            }
            for (String word : words) {
                if (!checkValid(word)) {
                    return "Neither";
                }
            }
            return "IPV4";
        }
        if (IP.contains(":")) {
            String[] words = IP.split(":");
            if (words.length != 8) {
                return "Neither";
            }
            for (String word : words) {
                if (!checkValidIPV6(word)) {
                    return "Neither";
                }
            }
            return "IPV6";
        }
        return "Neither";
    }

    private boolean checkValidIPV6(String word) {
        if (word == null || word.isEmpty()) {
            return false;
        }
        return word.indexOf("00") != 0;
    }

    public boolean IsContinuous(int[] numbers) {
        if (numbers == null || numbers.length == 0) {
            return false;
        }
        Arrays.sort(numbers);

        int zeroCount = 0;

        int min = 14;

        int max = -1;

        for (int i = 0; i < numbers.length; i++) {
            int number = numbers[i];
            if (number == 0) {
                zeroCount++;
                continue;
            }
            if (i > 0 && number == numbers[i - 1] - 1) {
                return false;
            }
            if (number > max) {
                max = number;
            }
            if (number < min) {
                min = number;
            }
        }
        if (zeroCount >= 4) {
            return true;
        }
        return max - min <= 4;
    }


    /**
     * @param root TreeNode类
     * @return int整型
     */
    public int maxDepth(TreeNode root) {
        // write code here
        if (root == null) {
            return 0;
        }
        return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
    }


    /**
     * @param n int整型
     * @param m int整型
     * @return int整型
     */
    public int ysf(int n, int m) {
        // write code here
        List<Integer> result = new ArrayList<>();
        for (int i = 1; i <= n; i++) {
            result.add(i);
        }
        int current = 0;

        while (result.size() > 1) {

            int index = (current + m - 1) % result.size();

            result.remove(index);

            current = index % result.size();
        }
        return result.get(0);
    }

    public ListNode mergeKLists(ArrayList<ListNode> lists) {
        if (lists == null || lists.isEmpty()) {
            return null;
        }
        PriorityQueue<ListNode> queue = new PriorityQueue<>(Comparator.comparingInt(o -> o.val));

        ListNode root = new ListNode(0);

        ListNode dummy = root;

        for (ListNode list : lists) {
            if (list != null) {
                queue.offer(list);
            }
        }
        while (!queue.isEmpty()) {
            ListNode node = queue.poll();

            dummy.next = node;

            dummy = dummy.next;

            if (node.next != null) {
                queue.offer(node.next);
            }
        }
        return root.next;
    }


    public ArrayList<String> Permutation(String str) {
        if (str == null) {
            return new ArrayList<>();
        }
        ArrayList<String> result = new ArrayList<>();
        char[] words = str.toCharArray();
        Arrays.sort(words);
        boolean[] used = new boolean[words.length];
        intervalPermute(result, words, "", used);
        return result;
    }

    private void intervalPermute(ArrayList<String> result, char[] words, String s, boolean[] used) {
        if (s.length() == words.length) {
            result.add(s);
            return;
        }
        for (int i = 0; i < words.length; i++) {
            if (used[i]) {
                continue;
            }
            if (i > 0 && words[i] == words[i - 1] && !used[i - 1]) {
                continue;
            }
            s += words[i];

            used[i] = true;

            intervalPermute(result, words, s, used);
            used[i] = false;
            s = s.substring(0, s.length() - 1);

        }
    }

    private void swap(char[] words, int i, int j) {
        char tmp = words[i];
        words[i] = words[j];
        words[j] = tmp;
    }


    /**
     * @param root TreeNode类 the root
     * @return bool布尔型一维数组
     */
    public boolean[] judgeIt(TreeNode root) {
        // write code here

        boolean[] result = new boolean[2];
        return null;
    }

    private boolean checkBalance(TreeNode root) {
        if (root == null) {
            return true;
        }
        int left = maxDepth(root.left);

        int right = maxDepth(root.right);

        return false;


    }

    /**
     * @param root TreeNode类
     * @return int整型
     */
    public int sumNumbers(TreeNode root) {
        // write code here
        if (root == null) {
            return 0;
        }
        return interval(root, 0);
    }

    private int interval(TreeNode root, int sum) {
        if (root == null) {
            return 0;
        }
        if (root.left == null && root.right == null) {
            return sum * 10 + root.val;
        }
        int val = sum * 10 + root.val;
        return interval(root.left, val) + interval(root.right, val);
    }


    public TreeNode reConstructBinaryTree(int[] pre, int[] in) {
        if (pre == null || in == null) {
            return null;
        }
        return reConstruct(0, pre, 0, in.length - 1, in);
    }

    private TreeNode reConstruct(int preStart, int[] pre, int inStart, int inEnd, int[] in) {
        if (preStart >= pre.length || inStart > inEnd) {
            return null;
        }
        TreeNode root = new TreeNode(pre[preStart]);
        int index = 0;
        for (int i = inStart; i <= inEnd; i++) {
            if (in[i] == root.val) {
                index = i;
                break;
            }
        }
        root.left = reConstruct(preStart + 1, pre, inStart, index - 1, in);

        root.right = reConstruct(preStart + index - inStart + 1, pre, index + 1, inEnd, in);

        return root;
    }


    public ListNode FindFirstCommonNode(ListNode pHead1, ListNode pHead2) {
        ListNode p1 = pHead1;

        ListNode p2 = pHead2;

        while (p1 != p2) {
            p1 = p1 == null ? pHead2 : p1.next;

            p2 = p2 == null ? pHead1 : p2.next;
        }
        return p1;
    }

    /**
     * @param str string字符串
     * @return int整型
     */
    public int atoi(String str) {
        // write code here
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

        if (words[index] == '-' || words[index] == '+') {
            sign = words[index] == '+' ? 1 : -1;
            index++;
        }
        long result = 0;
        while (index < words.length && Character.isDigit(words[index])) {
            result = result * 10 + Character.getNumericValue(words[index]);

            if (result > Integer.MAX_VALUE) {
                return sign == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
            }
            index++;
        }
        return (int) (sign * result);
    }


    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * <p>
     * 求出a、b的最大公约数。
     *
     * @param a int
     * @param b int
     * @return int
     */
    public int gcd(int a, int b) {
        // write code here
        if (b == 0) {
            return a;
        }
        return gcd(b, a % b);
    }


    /**
     * @param num int整型一维数组
     * @return TreeNode类
     */
    public TreeNode sortedArrayToBST(int[] num) {
        // write code here
        if (num == null || num.length == 0) {
            return null;
        }
        return intervalSort(0, num.length - 1, num);
    }

    private TreeNode intervalSort(int start, int end, int[] num) {
        if (start > end) {
            return null;
        }
        int mid = start + (end - start) / 2;
        TreeNode root = new TreeNode(num[mid]);
        root.left = intervalSort(start, mid - 1, num);
        root.right = intervalSort(mid + 1, end, num);
        return root;
    }


    /**
     * @param s string字符串
     * @return int整型
     */
    public int longestValidParentheses(String s) {
        // write code here
        if (s == null || s.isEmpty()) {
            return 0;
        }
        Stack<Integer> stack = new Stack<>();
        char[] words = s.toCharArray();
        int left = 0;
        int result = 0;
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
                    result = Math.max(result, i - left + 1);
                } else {
                    result = Math.max(result, i - stack.peek());
                }

            }
        }
        return result;
    }


    /**
     * 最少货币数
     *
     * @param arr int整型一维数组 the array
     * @param aim int整型 the target
     * @return int整型
     */
    public int minMoney(int[] arr, int aim) {
        // write code here
        int[] dp = new int[aim + 1];

        Arrays.fill(dp, aim + 1);

        for (int i = 1; i <= aim; i++) {
            for (int j = 0; j < arr.length; j++) {
                if (i - arr[j] >= 0) {
                    dp[i] = Math.min(dp[i], dp[i - arr[j]] + 1);
                }
            }
        }
        return dp[aim] == aim + 1 ? -1 : dp[aim];
    }


    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param nums   int整型一维数组
     * @param target int整型
     * @return int整型
     */
    public int searchSort(int[] nums, int target) {
        // write code here
        if (nums == null || nums.length == 0) {
            return -1;
        }
        int left = 0;
        int right = nums.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[left] <= nums[mid]) {
                if (target < nums[mid] && target >= nums[left]) {
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
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param pRoot TreeNode类
     * @return TreeNode类
     */
    public TreeNode Mirror(TreeNode pRoot) {
        // write code here
        if (pRoot == null) {
            return null;
        }
        TreeNode left = pRoot.left;

        TreeNode right = pRoot.right;

        pRoot.left = right;

        pRoot.right = left;

        Mirror(left);

        Mirror(right);

        return pRoot;
    }


    /**
     * @param root TreeNode类
     * @return bool布尔型
     */
    public boolean isSymmetric(TreeNode root) {
        // write code here
        if (root == null) {
            return true;
        }
        return intervalSymmetric(root.left, root.right);
    }

    private boolean intervalSymmetric(TreeNode left, TreeNode right) {
        if (left == null && right == null) {
            return true;
        }
        if (left == null || right == null) {
            return false;
        }
        if (left.val != right.val) {
            return false;
        }
        return intervalSymmetric(left.left, right.right) && intervalSymmetric(left.right, right.left);
    }


    /**
     * @param S string字符串
     * @param T string字符串
     * @return string字符串
     */
    public String minWindow(String S, String T) {
        // write code here
        if (S == null || T == null) {
            return "";
        }
        int n = T.length();
        int[] hash = new int[256];
        for (int i = 0; i < n; i++) {
            int index = T.charAt(i) - 'A';
            hash[index]++;
        }
        int result = Integer.MAX_VALUE;
        int beginIndex = 0;
        int head = 0;
        int m = S.length();
        int endIndex = 0;
        while (endIndex < m) {
            int index = S.charAt(endIndex++) - 'A';
            if (hash[index]-- > 0) {
                n--;
            }
            while (n == 0) {
                if (endIndex - beginIndex < result) {
                    head = beginIndex;
                    result = endIndex - beginIndex;
                }
                if (hash[S.charAt(beginIndex++) - 'A']++ < 0) {
                    n++;
                }
            }
        }
        if (result != Integer.MAX_VALUE) {
            return S.substring(head, head + result);
        }
        return "";
    }


    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param array int整型一维数组
     * @return int整型一维数组
     */
    public int[] FindNumsAppearOnce(int[] array) {
        // write code here
        if (array == null || array.length == 0) {
            return new int[]{};
        }
        if (array.length == 2) {
            return array;
        }
        int tmp = 0;
        for (int num : array) {
            tmp ^= num;
        }
        tmp &= -tmp;
        int[] result = new int[2];
        for (int num : array) {
            if ((tmp & num) != 0) {
                result[1] ^= num;
            } else {
                result[0] ^= num;
            }
        }
        return result;
    }


    public int GetNumberOfK(int[] array, int k) {
        if (array == null || array.length == 0) {
            return 0;
        }
        int left = 0;
        int right = array.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (array[mid] < k) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        if (array[left] != k) {
            return 0;
        }
        int firstIndex = left;
        right = array.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2 + 1;
            if (array[mid] > k) {
                right = mid - 1;
            } else {
                left = mid;
            }
        }
        int secondIndex = left;

        return secondIndex - firstIndex + 1;
    }


    /**
     * 最大数
     *
     * @param nums int整型一维数组
     * @return string字符串
     */
    public String solveBigString(int[] nums) {
        // write code here
        if (nums == null || nums.length == 0) {
            return "";
        }
        String[] params = new String[nums.length];

        for (int i = 0; i < nums.length; i++) {
            params[i] = String.valueOf(nums[i]);
        }
        Arrays.sort(params, (o1, o2) -> {
            String s1 = o1 + o2;
            String s2 = o2 + o1;
            return s2.compareTo(s1);
        });
        StringBuilder builder = new StringBuilder();
        for (String item : params) {
            if (!(builder.length() == 0 && "0".equals(item))) {
                builder.append(item);
            }
        }
        return builder.length() == 0 ? "0" : builder.toString();
    }

    public ArrayList<Integer> maxInWindows(int[] num, int size) {
        if (num == null || num.length == 0 || size == 0) {
            return new ArrayList<>();
        }
        LinkedList<Integer> linkedList = new LinkedList<>();

        ArrayList<Integer> result = new ArrayList<>();

        for (int i = 0; i < num.length; i++) {
            int index = i - size + 1;
            if (!linkedList.isEmpty() && linkedList.peekFirst() < index) {
                linkedList.pollFirst();
            }
            while (!linkedList.isEmpty() && num[linkedList.peekLast()] <= num[i]) {
                linkedList.pollLast();
            }
            linkedList.offer(i);
            if (index >= 0) {
                result.add(num[linkedList.peekFirst()]);
            }
        }
        return result;
    }


    /**
     * @param head ListNode类
     * @param m    int整型
     * @param n    int整型
     * @return ListNode类
     */
    public ListNode reverseBetween(ListNode head, int m, int n) {
        // write code here
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
        for (int i = 0; i < n; i++) {
            fast = fast.next;
        }
        ListNode begin = slow.next;

        ListNode end = fast.next;

        slow.next = reverse(begin, end);

        begin.next = end;

        return root.next;
    }


    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param head ListNode类
     * @return ListNode类
     */
    public ListNode oddEvenList(ListNode head) {
        // write code here
        if (head == null || head.next == null) {
            return head;
        }
        ListNode oddRoot = new ListNode(0);

        ListNode odd = oddRoot;

        ListNode evenRoot = new ListNode(0);

        ListNode even = evenRoot;
        int index = 1;
        while (head != null) {
            if (index % 2 == 1) {
                odd.next = head;
                odd = odd.next;
            } else {
                even.next = head;
                even = evenRoot.next;
            }
            head = head.next;

            index++;
        }
        even.next = null;
        odd.next = evenRoot.next;

        return oddRoot.next;
    }


    public ArrayList<ArrayList<Integer>> permuteUnique(int[] num) {
        if (num == null || num.length == 0) {
            return new ArrayList<>();
        }
        Arrays.sort(num);
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        boolean[] used = new boolean[num.length];
        intervalPermute(result, new ArrayList<Integer>(), used, num);
        return result;
    }

    private void intervalPermute(ArrayList<ArrayList<Integer>> result, List<Integer> tmp, boolean[] used, int[] num) {
        if (tmp.size() == num.length) {
            result.add(new ArrayList<>(tmp));
            return;
        }
        for (int i = 0; i < num.length; i++) {
            if (i > 0 && num[i] == num[i - 1] && !used[i - 1]) {
                continue;
            }
            if (used[i]) {
                continue;
            }
            used[i] = true;
            tmp.add(num[i]);
            intervalPermute(result, tmp, used, num);
            used[i] = false;
            tmp.remove(tmp.size() - 1);
        }
    }

    public ArrayList<Interval> merge(ArrayList<Interval> intervals) {
        if (intervals == null || intervals.isEmpty()) {
            return new ArrayList<>();
        }
        intervals.sort(Comparator.comparingInt(o -> o.start));

        ArrayList<Interval> result = new ArrayList<>();

        int index = 0;

        for (Interval interval : intervals) {
            if (result.isEmpty() || result.get(index - 1).end < interval.start) {
                result.add(interval);
                index++;
            } else {
                Interval pre = result.get(index - 1);
                pre.start = Math.min(pre.start, interval.start);
                pre.end = Math.max(pre.end, interval.end);
            }
        }
        return result;
    }


    /**
     * todo
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * <p>
     * 计算模板串S在文本串T中出现了多少次
     *
     * @param S string字符串 模板串
     * @param T string字符串 文本串
     * @return int整型
     */
    public int kmp(String S, String T) {
        // write code here
        return -1;
    }


    /**
     * @param grid int整型二维数组 the matrix
     * @return int整型
     */
    public int minPathSum(int[][] grid) {
        // write code here
        if (grid == null || grid.length == 0) {
            return 0;
        }
        int row = grid.length;

        int column = grid[0].length;

        int[] dp = new int[column];

        for (int j = 0; j < column; j++) {
            dp[j] = j == 0 ? grid[0][j] : dp[j - 1] + grid[0][j];
        }
        for (int i = 1; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (j == 0) {
                    dp[j] += grid[i][j];
                } else {
                    dp[j] = Math.min(dp[j], dp[j - 1]) + grid[i][j];
                }
            }
        }
        return dp[column - 1];
    }


    /**
     * 判断岛屿数量
     *
     * @param grid char字符型二维数组
     * @return int整型
     */
    public int solveIsland(char[][] grid) {
        // write code here
        if (grid == null || grid.length == 0) {
            return 0;
        }
        int row = grid.length;
        int column = grid[0].length;
        boolean[][] used = new boolean[row][column];
        int count = 0;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (grid[i][j] == '1') {
                    intervalDFS(grid, i, j, used);
                    count++;
                }
            }
        }
        return count;

    }

    private void intervalDFS(char[][] grid, int i, int j, boolean[][] used) {
        if (i < 0 || i >= used.length || j < 0 || j >= used[i].length || used[i][j]) {
            return;
        }
        if (grid[i][j] != '1') {
            return;
        }
        used[i][j] = true;
        grid[i][j] = '0';
        intervalDFS(grid, i - 1, j, used);
        intervalDFS(grid, i + 1, j, used);
        intervalDFS(grid, i, j - 1, used);
        intervalDFS(grid, i, j + 1, used);
    }


}
