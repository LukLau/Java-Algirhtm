package org.learn.algorithm.swordoffer;

import org.learn.algorithm.datastructure.Interval;
import org.learn.algorithm.datastructure.ListNode;
import org.learn.algorithm.datastructure.TreeNode;

import java.util.*;
import java.util.List;

/**
 * @author dora
 * @date 2021/6/10
 */
public class NormalSolution {
    /**
     * NC138 矩阵最长递增路径
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * 递增路径的最大长度
     *
     * @param matrix int整型二维数组 描述矩阵的每个数
     * @return int整型
     */

    private int incrementResult = 0;

    public static void main(String[] args) {
        NormalSolution solution = new NormalSolution();
        ListNode root = new ListNode(1);
//        solution.basicCalculate("100+100");
//        solution.minMoney(new int[]{5, 2, 3}, 20);
        int[] prices = new int[]{1, 2, 3, 0, 2};
        int profit = solution.maxProfitV(prices);
        System.out.println(profit);

    }

    /**
     * NC78 反转链表
     *
     * @param head
     * @return
     */
    public ListNode ReverseList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode prev = null;
        while (head != null) {
            ListNode node = head.next;
            head.next = prev;
            prev = head;
            head = node;
        }
        return prev;
    }

    public ListNode ReverseListV2(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode tmp = ReverseList(head.next);
        head.next.next = head;
        head.next = null;
        return tmp;
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
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (fast == slow) {
                return true;
            }
        }
        return false;
    }

    public ListNode EntryNodeOfLoop(ListNode pHead) {
        if (pHead == null || pHead.next == null) {
            return null;
        }
        ListNode fast = pHead;
        ListNode slow = pHead;
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (fast == slow) {
                fast = pHead;
                while (fast != slow) {
                    fast = fast.next;
                    slow = slow.next;
                }
                return fast;
            }
        }
        return null;
    }

    /**
     * NC48 在旋转过的有序数组中寻找目标值
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

    /**
     * WC87 最小的K个数
     *
     * @param input
     * @param k
     * @return
     */
    public ArrayList<Integer> GetLeastNumbers_Solution(int[] input, int k) {
        if (input == null || input.length == 0 || k <= 0) {
            return new ArrayList<>();
        }
        int index = getPartition(input, 0, input.length - 1);
        k--;
        while (index != k) {
            if (index > k) {
                index = getPartition(input, 0, index - 1);
            } else {
                index = getPartition(input, index + 1, input.length - 1);
            }
        }
        ArrayList<Integer> result = new ArrayList<>();
        for (int i = 0; i < k; i++) {
            result.add(input[i]);
        }
        return result;
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

    /**
     * NC22 合并两个有序的数组
     *
     * @param A
     * @param m
     * @param B
     * @param n
     */
    public void merge(int A[], int m, int B[], int n) {
        int k = m + n - 1;
        m--;
        n--;
        while (m >= 0 && n >= 0) {
            if (A[m] > B[n]) {
                A[k--] = A[m--];
            } else {
                A[k--] = B[n--];
            }
        }
        while (n >= 0) {
            A[k--] = B[n--];
        }
    }

    /**
     * NC88 寻找第K大
     *
     * @param a
     * @param n
     * @param K
     * @return
     */
    public int findKth(int[] a, int n, int K) {
        // write code here
        K--;
        int reverse = n - 1 - K;
        int index = getPartition(a, 0, a.length - 1);
        while (index != reverse) {
            if (index < reverse) {
                index = getPartition(a, index + 1, a.length - 1);
            } else {
                index = getPartition(a, 0, index - 1);
            }
        }
        return a[reverse];
    }

    public String solve(String str) {
        // write code here
        if (str == null || str.isEmpty()) {
            return "";
        }
        StringBuilder builder = new StringBuilder();
        return builder.append(str).reverse().toString();
    }

    /**
     * NC17 最长回文子串
     *
     * @param A
     * @return
     */
    public int getLongestPalindrome(String A) {
        // write code here
        if (A == null || A.length() == 0) {
            return 0;
        }
        int m = A.length();
        int n = A.length();
        boolean[][] dp = new boolean[m][n];
        int result = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j <= i; j++) {
                if (A.charAt(j) == A.charAt(i) && (i - j <= 2 || dp[j + 1][i - 1])) {
                    dp[j][i] = true;
                }
                if (dp[j][i] && i - j + 1 > result) {
                    result = i - j + 1;
                }
            }
        }
        return result;
    }

    /**
     * NC19 连续子数组的最大和
     *
     * @param array
     * @return
     */
    public int FindGreatestSumOfSubArray(int[] array) {
        if (array == null || array.length == 0) {
            return 0;
        }
        int local = 0;
        int result = Integer.MIN_VALUE;
        for (int i = 0; i < array.length; i++) {
            local = local > 0 ? local + array[i] : array[i];
            result = Math.max(result, local);
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
     * NC50 链表中的节点每k个一组翻转
     *
     * @param head
     * @param k
     * @return
     */
    public ListNode reverseKGroup(ListNode head, int k) {
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
            current = reverseKGroup(current, k);
            while (count-- > 0) {
                ListNode tmp = head.next;
                head.next = current;
                current = head;
                head = tmp;
            }
            head = current;
        }
        return head;
    }

    public ListNode reverseKGroupV2(ListNode head, int k) {
        if (head == null || head.next == null) {
            return head;
        }
        int count = 0;
        ListNode current = head;
        while (current != null && count != k) {
            count++;
            current = current.next;
        }
        if (count != k) {
            return head;
        }
        ListNode reverseListNode = reverseListNode(head, current);


        head.next = reverseKGroupV2(current, k);

        return reverseListNode;
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
     * BM49 表达式求值
     * 表达式默认值
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * 返回表达式的值
     *
     * @param s string字符串 待计算的表达式
     * @return int整型
     */
    public int basicCalculate(String s) {
        // write code here
        if (s == null || s.isEmpty()) {
            return 0;
        }
        int len = s.length();
        char sign = '+';
        int i = 0;
        Stack<Integer> stack = new Stack<>();
        while (i < len) {
            char tmp = s.charAt(i);
            if (tmp == '(') {
                int endIndex = i;
                int count = 0;
                while (endIndex < len) {
                    char endTmp = s.charAt(endIndex);
                    if (endTmp != '(' && endTmp != ')') {
                        endIndex++;
                        continue;
                    }
                    if (endTmp == '(') {
                        count++;
                    }
                    if (endTmp == ')') {
                        count--;
                    }
                    if (count == 0) {
                        break;
                    }
                    endIndex++;
                }
                String subString = s.substring(i + 1, endIndex);
                int value = basicCalculate(subString);
                stack.push(value);
                i = endIndex + 1;
            } else if (Character.isDigit(tmp)) {
                int val = 0;
                while (i < len && Character.isDigit(s.charAt(i))) {
                    val = val * 10 + Character.getNumericValue(s.charAt(i));
                    i++;
                }
                stack.push(val);
            }
            if (i == len || s.charAt(i) != ' ') {
                if (sign == '-') {
                    stack.push(-1 * stack.pop());
                }
                if (sign == '*') {
                    stack.push(stack.pop() * stack.pop());
                } else if (sign == '/') {
                    Integer secondValue = stack.pop();
                    Integer firstValue = stack.pop();
                    stack.push(firstValue / secondValue);
                }
                if (i != len) {
                    sign = s.charAt(i);
                }
            }
            i++;
        }
        int result = 0;
        for (Integer num : stack) {
            result += num;
        }
        return result;
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
        ListNode root = new ListNode(0);

        head1 = reverse(head1);

        head2 = reverse(head2);

        ListNode dummy = root;
        int carry = 0;
        while (head1 != null || head2 != null || carry != 0) {
            int val = (head1 == null ? 0 : head1.val) + (head2 == null ? 0 : head2.val) + carry;
            ListNode node = new ListNode(val % 10);

            carry = val / 10;
            dummy.next = node;
            dummy = dummy.next;
            head1 = head1 == null ? null : head1.next;
            head2 = head2 == null ? null : head2.next;
        }
        ListNode result = root.next;
        root.next = null;
        return reverse(result);
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
    public int maxProfitI(int[] prices) {
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

    /**
     * NC134 买卖股票的最好时机(二)
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * 计算最大收益
     *
     * @param prices int整型一维数组 股票每一天的价格
     * @return int整型
     */
    public int maxProfitMultiSell(int[] prices) {
        // write code here
        if (prices == null || prices.length == 0) {
            return 0;
        }
        int result = 0;
        int cost = prices[0];
        for (int i = 1; i < prices.length; i++) {
            int val = prices[i];
            if (val > cost) {
                result += val - cost;
            }
            cost = val;
        }
        return result;
    }

    /**
     * NC98 判断t1树中是否有与t2树完全相同的子树
     *
     * @param root1 TreeNode类
     * @param root2 TreeNode类
     * @return bool布尔型
     */
    public boolean isContains(TreeNode root1, TreeNode root2) {
        // write code here
        if (root1 == null && root2 == null) {
            return true;
        }
        if (root1 == null || root2 == null) {
            return false;
        }
        if (root1.val == root2.val) {
            return isContains(root1.left, root2.left) && isContains(root1.right, root2.right);
        }
        return false;
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
            int val = (m >= 0 ? Character.getNumericValue(s.charAt(m)) : 0) + (n >= 0 ? Character.getNumericValue(t.charAt(n)) : 0) + carry;

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
     * NC32 求平方根
     *
     * @param x int整型
     * @return int整型
     */
    public int sqrt(int x) {
        // write code here
        double precision = 0.00001;
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
            return "-1";
        }
        int m = s1.length();
        int n = s2.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (s1.charAt(i - 1) == s2.charAt(j - 1)) {
                    dp[i][j] = 1 + dp[i - 1][j - 1];
                } else {
                    dp[i][j] = Math.max(dp[i][j - 1], dp[i - 1][j]);
                }
            }
        }
        StringBuilder builder = new StringBuilder();
        while (m >= 1 && n >= 1) {
            if (s1.charAt(m - 1) == s2.charAt(n - 1)) {
                builder.append(s1.charAt(m - 1));
                m--;
                n--;
            } else if (dp[m - 1][n] > dp[m][n - 1]) {
                m--;
            } else {
                n--;
            }
        }
        return builder.length() == 0 ? "-1" : builder.reverse().toString();
    }

    /**
     * NC53 删除链表的倒数第n个节点
     *
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
        ListNode fast = head;
        while (fast.next != null) {
            fast = fast.next;
            count++;
        }
        ListNode root = new ListNode(0);
        root.next = head;

        fast = root;


        for (int i = 0; i < count - n; i++) {
            fast = fast.next;
        }
        fast.next = fast.next.next;

        return root.next;
    }

    public ListNode removeNthFromEndII(ListNode head, int n) {
        if (head == null) {
            return null;
        }
        ListNode root = new ListNode(0);
        root.next = head;

        ListNode fast = root;
        for (int i = 0; i < n; i++) {
            fast = fast.next;
        }
        ListNode slow = root;

        while (fast.next != null) {
            slow = slow.next;
            fast = fast.next;
        }
        slow.next = slow.next.next;
        return root.next;
    }

    /**
     * 判断岛屿数量
     *
     * @param grid char字符型二维数组
     * @return int整型
     */
    public int islandNum(char[][] grid) {
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
                if (used[i][j]) {
                    continue;
                }
                if (grid[i][j] == '1') {
                    count++;
                    internalIsland(grid, i, j, used);

                }
            }
        }
        return count;
    }

    private void internalIsland(char[][] words, int i, int j, boolean[][] used) {
        if (i < 0 || i == words.length || j < 0 || j == words[i].length) {
            return;
        }
        if (used[i][j]) {
            return;
        }
        used[i][j] = true;
        if (words[i][j] == '0') {
            return;
        }
        internalIsland(words, i - 1, j, used);
        internalIsland(words, i + 1, j, used);
        internalIsland(words, i, j - 1, used);
        internalIsland(words, i, j + 1, used);
    }


    /**
     * @param k    int整型
     * @return ListNode类
     */

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
     * NC97 字符串出现次数的TopK问题
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

        for (String item : strings) {
            Integer count = map.getOrDefault(item, 0);
            map.put(item, count + 1);
        }
        PriorityQueue<String> priorityQueue = new PriorityQueue<>(k, (o1, o2) -> {
            Integer count1 = map.getOrDefault(o1, 0);
            Integer count2 = map.getOrDefault(o2, 0);
            if (count1.equals(count2)) {
                return o2.compareTo(o1);
            }
            return count1.compareTo(count2);
        });
        for (Map.Entry<String, Integer> item : map.entrySet()) {
            priorityQueue.offer(item.getKey());
            if (priorityQueue.size() > k) {
                priorityQueue.poll();
            }
        }
        String[][] result = new String[k][2];
        for (int i = 0; i < k; i++) {
            String item = priorityQueue.poll();
            result[k - 1 - i][0] = item;
            result[k - 1 - i][1] = map.get(item).toString();
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
     * NC70 单链表的排序
     *
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
        ListNode second = slow.next;
        slow.next = null;
        ListNode first = sortInList(head);
        ListNode newSecond = sortInList(second);

        return sort(first, newSecond);
    }

    private ListNode sort(ListNode l1, ListNode l2) {
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
            l1.next = sort(l1.next, l2);
            return l1;
        } else {
            l2.next = sort(l1, l2.next);
            return l2;
        }
    }

    /**
     * NC96 判断一个链表是否为回文结构
     *
     * @param head ListNode类 the head
     * @return bool布尔型
     */
    public boolean isPail(ListNode head) {
        // write code here
        if (head == null) {
            return false;
        }
        ListNode fast = head;
        ListNode slow = head;
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        ListNode newNode = slow.next;
        slow.next = null;
        ListNode reverse = reverse(newNode);
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
     * NC30 缺失的第一个正整数
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param nums int整型一维数组
     * @return int整型
     */
    public int minNumberDisappeared(int[] nums) {
        // write code here
        if (nums == null || nums.length == 0) {
            return -1;
        }
        for (int i = 0; i < nums.length; i++) {
            while (nums[i] > 0 && nums[i] < nums.length && nums[i] != nums[nums[i] - 1]) {
                swap(nums, i, nums[i] - 1);
            }
        }
        for (int i = 0; i < nums.length; i++) {
            if (i + 1 != nums[i]) {
                return i + 1;

            }
        }
        return nums.length + 1;
    }

    /**
     * NC31 第一个只出现一次的字符
     *
     * @param str
     * @return
     */
    public int FirstNotRepeatingChar(String str) {
        int[] hash = new int[512];
        char[] words = str.toCharArray();
        for (int i = 0; i < words.length; i++) {
            hash[words[i]]++;
        }
        for (int i = 0; i < words.length; i++) {
            if (hash[words[i]] == 1) {
                return i;
            }
        }
        return -1;
    }

    /**
     * todo
     * NC133 链表的奇偶重排
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
        return null;
    }

    /**
     * @param head ListNode类
     * @return ListNode类
     */
    public ListNode deleteDuplicatesII(ListNode head) {
        // write code here
        if (head == null || head.next == null) {
            return head;
        }
        if (head.val == head.next.val) {
            return deleteDuplicatesII(head.next);
        }
        head.next = deleteDuplicatesII(head.next);
        return head;
    }

    /**
     * BM16 删除有序链表中重复的元素-II
     *
     * @param head
     * @return
     */
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        if (head.val == head.next.val) {
            ListNode current = head.next;
            while (current != null && current.val == head.val) {
                current = current.next;
            }
            return deleteDuplicates(current);
        }
        head.next = deleteDuplicates(head.next);
        return head;
    }


    /**
     * NC26 括号生成
     *
     * @param n int整型
     * @return string字符串ArrayList
     */
    public ArrayList<String> generateParenthesis(int n) {
        // write code here
        if (n <= 0) {
            return new ArrayList<>();
        }
        ArrayList<String> result = new ArrayList<>();
        internalGenerateParenthesis(result, 0, 0, n, "");
        return result;
    }

    private void internalGenerateParenthesis(ArrayList<String> result, int open, int close, int n, String s) {
        if (s.length() == 2 * n) {
            result.add(s);
            return;
        }
        if (open < n) {
            internalGenerateParenthesis(result, open + 1, close, n, s + "(");
        }
        if (close < open) {
            internalGenerateParenthesis(result, open, close + 1, n, s + ")");
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
     * NC135 买卖股票的最好时机(三)
     *
     * @param prices
     * @return
     */
    public int maxProfitThree(int[] prices) {
        if (prices == null || prices.length == 0) {
            return 0;
        }
        int[] leftSell = new int[prices.length];
        int cost = prices[0];
        int leftResult = 0;
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > cost) {
                leftResult = Math.max(leftResult, prices[i] - cost);
            } else {
                cost = prices[i];
            }
            leftSell[i] = leftResult;
        }
        int[] rightSell = new int[prices.length + 1];
        int rightResult = 0;

        cost = prices[prices.length - 1];

        for (int i = prices.length - 2; i >= 0; i--) {
            if (prices[i] < cost) {
                rightResult = Math.max(rightResult, cost - prices[i]);
            } else {
                cost = prices[i];
            }
            rightSell[i] = rightResult;
        }
        int result = 0;
        for (int i = leftSell.length - 1; i >= 0; i--) {
            result = Math.max(result, leftSell[i] + rightSell[i]);
        }
        return result;
    }

    /**
     * NC138 矩阵最长递增路径
     *
     * @param matrix
     * @return
     */
    public int longestIncrementMatrix(int[][] matrix) {
        // write code here
        if (matrix == null || matrix.length == 0) {
            return 0;
        }
        int row = matrix.length;
        int column = matrix[0].length;
        int result = 0;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                int tmp = internalIncrementResult(i, j, matrix, Integer.MIN_VALUE, 0);
                result = Math.max(result, tmp);
            }
        }
        return result;
    }

    private int internalIncrementResult(int i, int j, int[][] matrix, int minValue, int result) {
        if (i < 0 || i >= matrix.length || j < 0 || j >= matrix[i].length) {
            return result;
        }
        if (matrix[i][j] <= minValue) {
            return result;
        }
        int count = 0;
        count = Math.max(count, internalIncrementResult(i - 1, j, matrix, matrix[i][j], result + 1));
        count = Math.max(count, internalIncrementResult(i + 1, j, matrix, matrix[i][j], result + 1));
        count = Math.max(count, internalIncrementResult(i, j - 1, matrix, matrix[i][j], result + 1));
        count = Math.max(count, internalIncrementResult(i, j + 1, matrix, matrix[i][j], result + 1));
        return count;
    }


    /**
     * NC129 阶乘末尾0的数量
     * the number of 0
     *
     * @param n long长整型 the number
     * @return long长整型
     */
    public long thenumberof0(long n) {
        // write code here
        if (n < 5) {
            return 0;
        }
        long count = 0;
        while (n / 5 != 0) {
            count += n / 5;
            n /= 5;
        }
        return count;
    }


    /**
     * todo
     * NC142 最长重复子串
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param a string字符串 待计算字符串
     * @return int整型
     */
    public int longestRepeatString(String a) {
        // write code here
        if (a == null || a.isEmpty()) {
            return 0;
        }
        LinkedList<String> linkedList = new LinkedList<>();
        linkedList.offer(a);
        while (!linkedList.isEmpty()) {
            String tmp = linkedList.pollFirst();
//            if (validRepeatString(t))
        }
        return -1;

    }


    /**
     * NC144 不相邻最大子序列和
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * 计算
     *
     * @param n     int整型 数组的长度
     * @param array int整型一维数组 长度为n的数组
     * @return long长整型
     */
    public long maxSubsequence(int n, int[] array) {
        // write code here
        if (array == null || array.length == 0) {
            return 0;
        }
        long prev = 0;
        long current = array[0];
        for (int i = 1; i < array.length; i++) {
            long tmp = current;
            current = Math.max(current, prev + array[i]);
            prev = tmp;
        }
        return Math.max(0, Math.max(prev, current));
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


    /**
     * NC57 反转数字
     *
     * @param x
     * @return
     */
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
     * NC58 找到搜索二叉树中两个错误的节点
     *
     * @param root TreeNode类 the root
     * @return int整型一维数组
     */
    public int[] findError(TreeNode root) {
        // write code here
        if (root == null) {
            return null;
        }
        Stack<TreeNode> stack = new Stack<>();
        TreeNode prev = null;
        TreeNode first = null;
        TreeNode second = null;
        TreeNode p = root;
        while (!stack.isEmpty() || p != null) {
            while (p != null) {
                stack.push(p);
                p = p.left;
            }
            p = stack.pop();
            if (prev != null) {
                if (first == null && prev.val >= p.val) {
                    first = prev;
                }
                if (first != null && prev.val >= p.val) {
                    second = p;
                }
            }
            prev = p;
            p = p.right;
        }
        if (first == null) {
            return null;
        }
        return new int[]{second.val, first.val};
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

                int val = Character.getNumericValue(s.charAt(i)) * Character.getNumericValue(t.charAt(j)) + pos[i + j + 1];

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

    /**
     * NC71 旋转数组的最小数字
     *
     * @param array
     * @return
     */
    public int minNumberInRotateArray(int[] array) {
        if (array == null || array.length == 0) {
            return -1;
        }
        int left = 0;
        int right = array.length - 1;
        while (left < right) {
            if (array[left] == array[right]) {
                right--;
            }
            int mid = left + (right - left) / 2;
            if (array[mid] > array[right]) {
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

    /**
     * 309. Best Time to Buy and Sell Stock with Cooldown
     * https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/
     * process think:
     * https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/discuss/75931/Easiest-JAVA-solution-with-explanations
     *
     * @param prices
     * @return
     */
    public int maxProfitV(int[] prices) {
        if (prices == null || prices.length <= 1) {
            return 0;
        }
        if (prices.length == 2) {
            return Math.max(0, prices[1] - prices[0]);
        }
        int len = prices.length;
        int[] buy = new int[len];
        int[] sell = new int[len];
        buy[0] = -prices[0];
        sell[1] = Math.max(0, prices[1] - prices[0]);
        buy[1] = Math.max(-prices[0], -prices[1]);
        for (int i = 2; i < len; i++) {
            buy[i] = Math.max(buy[i - 1], sell[i - 2] - prices[i]);
            sell[i] = Math.max(sell[i - 1], buy[i - 1] + prices[i]);
        }
        return sell[len - 1];
    }


    public int maxProfitVVariant(int[] prices) {
        if (prices == null || prices.length <= 1) {
            return 0;
        }
        int b0 = -prices[0];
        int b1 = b0;
        int s0 = 0;
        int s1 = 0;
        int s2 = 0;
        int len = prices.length;
        for (int i = 1; i < len; i++) {
            b0 = Math.max(b1, s2 - prices[i]);
            s0 = Math.max(s1, b1 + prices[i]);

            b1 = b0;
            s2 = s1;
            s1 = s0;
        }
        return s0;
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
     * NC23 划分链表
     *
     * @param head ListNode类
     * @param x    int整型
     * @return ListNode类
     */
    public ListNode partition(ListNode head, int x) {
        // write code here
        if (head == null || head.next == null) {
            return head;
        }
        ListNode s1 = new ListNode(0);
        ListNode s2 = new ListNode(0);

        ListNode fast = s1;
        ListNode slow = s2;
        while (head != null) {
            if (head.val <= x) {
                slow.next = head;
                slow = slow.next;
            } else {
                fast.next = head;
                fast = fast.next;
            }
            head = head.next;
        }
        slow.next = s1.next;
        fast.next = null;
        return s2.next;
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
        ListNode slow = root;
        for (int i = 0; i < k; i++) {
            fast = fast.next;
        }
        if (fast == null) {
            return null;
        }
        while (fast.next != null) {
            fast = fast.next;
            slow = slow.next;
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
     * NC20 数字字符串转化成IP地址
     *
     * @param s string字符串
     * @return string字符串ArrayList
     */
    public ArrayList<String> restoreIpAddresses(String s) {
        // write code here
        if (s == null || s.isEmpty()) {
            return new ArrayList<>();
        }
        ArrayList<String> result = new ArrayList<>();
        int n = s.length();
        for (int i = 1; i < 4 && i < n - 2; i++) {
            for (int j = i + 1; j < j + 4 && j < n - 1; j++) {
                for (int k = j + 1; k < k + 4 && k < n; k++) {
                    String a = s.substring(0, i);
                    String b = s.substring(i, j);
                    String c = s.substring(j, k);
                    String d = s.substring(k);
                    if (checkIp(a) && checkIp(b) && checkIp(c) && checkIp(d)) {
                        result.add(a + "." + b + "." + c + "." + d);
                    }
                }
            }
        }
        return result;
    }

    private boolean checkIp(String word) {
        if (word == null || word.length() == 0) {
            return false;
        }
        int num = Integer.parseInt(word);
        if (num < 0 || num > 255) {
            return false;
        }
        int len = word.length();
        return len <= 1 || word.charAt(0) != '0';
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

    /**
     * NC27 集合的所有子集(一)
     *
     * @param S
     * @return
     */
    public ArrayList<ArrayList<Integer>> subsets(int[] S) {
        if (S == null || S.length == 0) {
            return new ArrayList<>();
        }
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        internalSubSets(result, new ArrayList<>(), 0, S);
        return result;

    }

    private void internalSubSets(ArrayList<ArrayList<Integer>> result, ArrayList<Integer> tmp, int start, int[] s) {
        result.add(new ArrayList<>(tmp));
        for (int i = start; i < s.length; i++) {
            tmp.add(s[i]);
            internalSubSets(result, tmp, i + 1, s);
            tmp.remove(tmp.size() - 1);
        }
    }


    /**
     * NC56 回文数字
     *
     * @param x int整型
     * @return bool布尔型
     */
    public boolean isPalindrome(int x) {
        // write code here
        if (x < 0) {
            return false;
        }
        long result = 0;
        while (x > result) {
            result = result * 10 + x % 10;
            x /= 10;
        }
        return result == x || result / 10 == x;

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
        if (obstacleGrid[0][0] == 0) {
            dp[0] = 1;
        }
        for (int j = 1; j < column; j++) {
            dp[j] = obstacleGrid[0][j] == 1 ? 0 : dp[j - 1];
        }
        for (int i = 1; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (obstacleGrid[i][j] == 1) {
                    dp[j] = 0;
                } else {
                    dp[j] += j == 0 ? 0 : dp[j - 1];
                }
            }
        }
        return dp[column - 1];
    }

    /**
     * NC83 子数组最大乘积
     *
     * @param arr
     * @return
     */
    public double maxProduct(double[] arr) {
        double max = arr[0];
        double min = arr[0];
        double result = arr[0];
        for (int i = 1; i < arr.length; i++) {
            double tmpMax = Math.max(Math.max(max * arr[i], min * arr[i]), arr[i]);
            double tmpMin = Math.min(Math.min(min * arr[i], max * arr[i]), arr[i]);
            result = Math.max(tmpMax, result);
            max = tmpMax;
            min = tmpMin;
        }
        return result;
    }


    /**
     * NC116 把数字翻译成字符串
     *
     * @param nums string字符串 数字串
     * @return int整型
     */
    public int decodeString(String nums) {
        // write code here
        if (nums == null || nums.isEmpty()) {
            return 0;
        }
        int len = nums.length();
        int[] dp = new int[len + 1];
        dp[0] = 1;
        for (int i = 1; i <= len; i++) {
            String s1 = nums.substring(i - 1, i);
            int num1 = Integer.parseInt(s1);
            if (num1 >= 1 && num1 <= 9) {
                dp[i] += dp[i - 1];
            }
            if (i > 1) {
                String s2 = nums.substring(i - 2, i);
                int num2 = Integer.parseInt(s2);
                if (num2 >= 10 && num2 <= 26) {
                    dp[i] += dp[i - 2];
                }
            }
        }
        return dp[len];
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

    /**
     * WC93 数组中出现次数超过一半的数字
     *
     * @param array
     * @return
     */
    public int MoreThanHalfNum_Solution(int[] array) {
        int candidate = array[0];
        int count = 0;
        for (int num : array) {
            if (num == candidate) {
                count++;
                continue;
            }
            count--;
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
        return candidate;
    }

    /**
     * NC46 加起来和为目标值的组合(二)
     *
     * @param num
     * @param target
     * @return
     */
    public ArrayList<ArrayList<Integer>> combinationSum2(int[] num, int target) {
        if (num == null || num.length == 0) {
            return new ArrayList<>();
        }
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        boolean[] used = new boolean[num.length];
        Arrays.sort(num);
        internalCombinationSum2(result, new ArrayList<>(), 0, num, target, used);
        return result;
    }

    private void internalCombinationSum2(ArrayList<ArrayList<Integer>> result, ArrayList<Integer> tmp, int start, int[] num, int target, boolean[] used) {
        if (target == 0) {
            result.add(new ArrayList<>(tmp));
            return;
        }
        for (int i = start; i < num.length && num[i] <= target; i++) {
            if (used[i]) {
                continue;
            }
            if (i > start && num[i] == num[i - 1] && !used[i - 1]) {
                continue;
            }
            used[i] = true;
            tmp.add(num[i]);
            internalCombinationSum2(result, tmp, i + 1, num, target - num[i], used);
            used[i] = false;
            tmp.remove(tmp.size() - 1);
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
//                if (!checkValid(word)) {
//                    return "Neither";
//                }
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

    /**
     * NC63 扑克牌顺子
     *
     * @param numbers
     * @return
     */
    public boolean IsContinuous(int[] numbers) {
        if (numbers == null || numbers.length == 0) {
            return false;
        }
        Arrays.sort(numbers);
        int zeroCount = 0;
        int max = -1;
        int min = 14;
        for (int i = 0; i < numbers.length; i++) {
            int val = numbers[i];
            if (val == 0) {
                zeroCount++;
                continue;
            }
            if (i > 0 && numbers[i] == numbers[i - 1]) {
                return false;
            }
            if (val > max) {
                max = val;
            }
            if (val < min) {
                min = val;
            }
            if (max - min > 4) {
                return false;
            }
        }
        if (zeroCount >= 4) {
            return true;
        }
        return max - min <= 5;
    }

    public boolean IsContinuousV2(int[] numbers) {
        if (numbers == null || numbers.length == 0) {
            return false;
        }
        int result = 0;
        int min = 14;
        int max = -1;
        for (int val : numbers) {
            if (val == 0) {
                continue;
            }
            if ((1 << val & result) != 0) {
                return false;
            }
            if (val > max) {
                max = val;
            }
            if (val < min) {
                min = val;
            }
            if (max - min > 4) {
                return false;
            }
            result = 1 << val | result;
        }
        return true;
    }


    /**
     * NC13 二叉树的最大深度
     *
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

    /**
     * NC51 合并k个已排序的链表
     *
     * @param lists
     * @return
     */
    public ListNode mergeKLists(ArrayList<ListNode> lists) {
        if (lists == null || lists.isEmpty()) {
            return null;
        }
        PriorityQueue<ListNode> priorityQueue = new PriorityQueue<>(new Comparator<ListNode>() {
            @Override
            public int compare(ListNode o1, ListNode o2) {
                return o1.val - o2.val;
            }
        });
        for (ListNode node : lists) {
            if (node != null) {
                priorityQueue.offer(node);
            }
        }
        ListNode root = new ListNode(0);
        ListNode dummy = root;
        while (!priorityQueue.isEmpty()) {
            ListNode node = priorityQueue.poll();
            dummy.next = node;
            dummy = dummy.next;

            if (node.next != null) {
                priorityQueue.offer(node.next);
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


    /**
     * NC12 重建二叉树
     *
     * @param pre
     * @param in
     * @return
     */
    public TreeNode reConstructBinaryTree(int[] pre, int[] in) {
        if (pre == null || in == null) {
            return null;
        }
        return internalReConstruct(0, pre, 0, in.length - 1, in);
    }

    private TreeNode internalReConstruct(int preStart, int[] pre, int inStart, int inEnd, int[] in) {
        if (inStart > inEnd) {
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
        root.left = internalReConstruct(preStart + 1, pre, inStart, index - 1, in);
        root.right = internalReConstruct(preStart + index - inStart + 1, pre, index + 1, inEnd, in);
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
     * NC11 将升序数组转化为平衡二叉搜索树
     *
     * @param num int整型一维数组
     * @return TreeNode类
     */
    public TreeNode sortedArrayToBST(int[] num) {
        // write code here
        if (num == null || num.length == 0) {
            return null;
        }
        return internalSortedArrayToBST(num, 0, num.length - 1);
    }

    private TreeNode internalSortedArrayToBST(int[] num, int start, int end) {
        if (start > end) {
            return null;
        }
        int mid = start + (end - start) / 2;
        TreeNode root = new TreeNode(num[mid]);
        root.left = internalSortedArrayToBST(num, start, mid - 1);
        root.right = internalSortedArrayToBST(num, mid + 1, end);
        return root;
    }


    /**
     * NC49 最长的括号子串
     *
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
        int result = 0;
        int left = -1;
        for (int i = 0; i < words.length; i++) {
            if (words[i] == '(') {
                stack.push(i);
            } else {
                if (!stack.isEmpty()) {
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

    public int longestValidParenthesesV2(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        }
        Stack<Integer> stack = new Stack<>();
        char[] words = s.toCharArray();
        for (int i = 0; i < words.length; i++) {
            if (words[i] == '(' || stack.isEmpty()) {
                stack.push(i);
            } else if (words[stack.peek()] == '(') {
                stack.pop();
            } else {
                stack.push(i);
            }
        }
        int len = s.length();
        if (stack.isEmpty()) {
            return len;
        }
        int result = 0;
        int a = len;
        while (!stack.isEmpty()) {
            int b = stack.pop();
            result = Math.max(result, a - b - 1);
            a = b;
        }
        result = Math.max(result, a);
        return result;
    }


    /**
     * NC126 兑换零钱(一)
     * 最少货币数
     *
     * @param arr int整型一维数组 the array
     * @param aim int整型 the target
     * @return int整型
     */
    public int minMoney(int[] arr, int aim) {
        // write code here
        if (arr == null || arr.length == 0) {
            return 0;
        }
        int[] dp = new int[aim + 1];
        Arrays.fill(dp, Integer.MAX_VALUE);
        dp[0] = 0;
        for (int i = 1; i <= aim; i++) {
            int min = Integer.MAX_VALUE;
            for (int j = 0; j < arr.length; j++) {
                if (i - arr[j] >= 0 && dp[i - arr[j]] != Integer.MAX_VALUE) {
                    min = Math.min(min, 1 + dp[i - arr[j]]);
                }
            }
            dp[i] = min;
        }
        return dp[aim] == Integer.MAX_VALUE ? -1 : dp[aim];
    }

    /**
     * NC107 寻找峰值
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param nums int整型一维数组
     * @return int整型
     */
    public int findPeakElement(int[] nums) {
        // write code here
        if (nums == null || nums.length == 0) {
            return -1;
        }
        int left = 0;
        int right = nums.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < nums[mid + 1]) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return left;
    }


    /**
     * 最大正方形
     *
     * @param matrix char字符型二维数组
     * @return int整型
     */
    public int maximumSquare(char[][] matrix) {
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
                char tmp = matrix[i - 1][j - 1];
                if (tmp == '1') {
                    dp[i][j] = 1 + Math.min(Math.min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]);
                }
                if (dp[i][j] > 0) {
                    result = Math.max(result, dp[i][j] * dp[i][j]);
                }
            }
        }
        return result;
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
     * NC28 最小覆盖子串
     *
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
        int[] hash = new int[512];
        for (int i = 0; i < n; i++) {
            hash[T.charAt(i)]++;
        }
        int m = S.length();
        int rightSide = 0;
        int leftSide = 0;
        int result = Integer.MAX_VALUE;
        int head = 0;
        while (rightSide < m) {
            if (hash[S.charAt(rightSide++)]-- > 0) {
                n--;
            }
            while (n == 0) {
                if (rightSide - leftSide < result) {
                    head = leftSide;
                    result = Math.min(result, rightSide - leftSide);
                }
                if (hash[S.charAt(leftSide++)]++ == 0) {
                    n++;
                }
            }
        }
        if (result == Integer.MAX_VALUE) {
            return "";
        }
        return S.substring(head, head + result);

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
        int result = 0;
        for (int num : array) {
            result ^= num;
        }
        result &= -result;
        int[] tmp = new int[2];
        for (int num : array) {
            if ((result & num) != 0) {
                tmp[1] ^= num;
            } else {
                tmp[0] ^= num;
            }
        }
        Arrays.sort(tmp);
        return tmp;
    }


    /**
     * NC74 数字在升序数组中出现的次数
     *
     * @param array
     * @param k
     * @return
     */
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
        return left - firstIndex + 1;
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
     * NC21 链表内指定区间反转
     *
     * @param head ListNode类
     * @param m    int整型
     * @param n    int整型
     * @return ListNode类
     */
    public ListNode reverseBetween(ListNode head, int m, int n) {
        ListNode root = new ListNode(0);
        root.next = head;


        // write code here
        ListNode slow = root;

        ListNode fast = root;

        for (int i = 0; i < m - 1; i++) {
            slow = slow.next;
        }
        for (int i = 0; i < n; i++) {
            fast = fast.next;
        }

        ListNode start = slow.next;

        ListNode end = fast.next;

        slow.next = reverseListNode(start, end);

        start.next = end;

        return root.next;
    }

    private ListNode reverseListNode(ListNode start, ListNode end) {
        ListNode prev = start;
        while (start != end) {
            ListNode next = start.next;
            start.next = prev;
            prev = start;
            start = next;
        }
        return prev;
    }


    /**
     * NC42 有重复项数字的全排列
     *
     * @param num
     * @return
     */
    public ArrayList<ArrayList<Integer>> permuteUnique(int[] num) {
        if (num == null || num.length == 0) {
            return new ArrayList<>();
        }
        Arrays.sort(num);
        boolean[] used = new boolean[num.length];
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        internalPermuteUnique(result, new ArrayList<>(), num, used);
        return result;
    }

    private void internalPermuteUnique(ArrayList<ArrayList<Integer>> result, ArrayList<Integer> tmp, int[] num, boolean[] used) {
        if (tmp.size() == num.length) {
            result.add(new ArrayList<>(tmp));
            return;
        }
        for (int i = 0; i < num.length; i++) {
            if (used[i]) {
                continue;
            }
            if (i > 0 && num[i] == num[i - 1] && !used[i - 1]) {
                continue;
            }
            used[i] = true;
            tmp.add(num[i]);
            internalPermuteUnique(result, tmp, num, used);
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
     * NC59 矩阵的最小路径和
     *
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
     * NC60 判断一棵二叉树是否为搜索二叉树和完全二叉树
     *
     * @param root TreeNode类 the root
     * @return bool布尔型一维数组
     */
    public boolean[] judgeIt(TreeNode root) {
        // write code here
        boolean[] result = new boolean[2];
        result[0] = checkBST(root);
        result[1] = checkCBT(root);
        return result;
    }

    private boolean checkCBT(TreeNode root) {
        if (root == null) {
            return true;
        }
        if (root.left == null && root.right == null) {
            return true;
        }
        return root.left != null && checkCBT(root.left) && checkCBT(root.right);
    }

    private boolean checkBST(TreeNode root) {
        if (root == null) {
            return true;
        }
        return internalBST(Integer.MIN_VALUE, root, Integer.MAX_VALUE);
    }

    private boolean internalBST(int minValue, TreeNode root, int maxValue) {
        if (root == null) {
            return true;
        }
        if (root.val <= minValue || root.val >= maxValue) {
            return false;
        }
        return internalBST(minValue, root.left, root.val) && internalBST(root.val, root.right, maxValue);
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
