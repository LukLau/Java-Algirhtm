package org.dora.algorithm.solution.v2;

import javafx.scene.transform.Rotate;
import org.dora.algorithm.datastructe.*;

import java.util.*;

/**
 * @author dora
 * @date 2019/9/30
 */
public class ThreeHundred {


    public static void main(String[] args) {
        ThreeHundred threeHundred = new ThreeHundred();
        String num = "232";
        int target = 8;

        //        List<String> result = threeHundred.addOperators(num, target);
        //        System.out.println(result);
        //        int[][] edges = new int[][]{{3, 0}, {3, 1}, {3, 2}, {3, 4}, {5, 4}};
        //        threeHundred.findMinHeightTrees(6, edges);

        Point point1 = new Point(0, 0);
        Point point2 = new Point(0, 1);
        Point point3 = new Point(2, 2);
        Point point4 = new Point(2, 2);
        //        Point point5 = new Point();
        List<Point> pointList = Arrays.asList(point1, point2, point3, point4);

        //        threeHundred.numIslands2(3, 3, pointList.toArray(new Point[]{}));

        // TreeNode root = new TreeNode(1);
        // TreeNode left = new TreeNode(2);
        // left.left = new TreeNode(3);
        // root.left = left;

        // threeHundred.verticalOrder(root);
        // threeHundred.removeDuplicateLetters("bcabc");
        String[] words = new String[]{"a", "ab", "abc", "d", "cd", "bcd", "abcd"};
        // int result = threeHundred.maxProduct(words);
        // System.out.println(result);
        // threeHundred.bulbSwitch(99);
        // threeHundred.generateAbbreviations("word");
        int[] coins = new int[]{2, 5, 5};
        int[][] edges = new int[][]{{1, 67}, {2, 34}, {4, 69}, {5, 24}, {6, 78}, {7, 58}, {8, 62}, {9, 64}, {10, 5}, {11, 45}, {12, 81}, {13, 27}, {14, 61}, {15, 91}, {16, 95}, {17, 42}, {18, 27}, {19, 36}, {20, 91}, {21, 4}, {22, 2}, {23, 53}, {24, 92}, {25, 82}, {26, 21}, {27, 16}, {28, 18}, {29, 95}, {30, 47}, {31, 26}, {32, 71}, {33, 38}, {34, 69}, {35, 12}, {36, 67}, {37, 99}, {38, 35}, {39, 94}, {40, 3}, {41, 11}, {42, 22}, {43, 33}, {44, 73}, {45, 64}, {46, 41}, {47, 11}, {48, 53}, {49, 68}, {50, 47}, {51, 44}, {52, 62}, {53, 57}, {54, 37}, {55, 59}, {56, 23}, {57, 41}, {58, 29}, {59, 78}, {60, 16}, {61, 35}, {62, 90}, {63, 42}, {64, 88}, {65, 6}, {66, 40}, {67, 42}, {68, 64}, {69, 48}, {70, 46}, {71, 5}, {72, 90}, {73, 29}, {74, 70}, {75, 50}, {76, 6}, {77, 1}, {78, 93}, {79, 48}, {80, 29}, {81, 23}, {82, 84}, {83, 54}, {84, 56}, {85, 40}, {86, 66}, {87, 76}, {88, 31}, {89, 8}, {90, 44}, {91, 39}, {92, 26}, {93, 23}, {94, 37}, {95, 38}, {96, 18}, {97, 82}, {98, 29}, {99, 41}};
        // threeHundred.countComponents(100, edges);
        int[][] increase = new int[][]{{9, 9, 4}, {6, 6, 8}, {2, 1, 1}};
        // threeHundred.longestIncreasingPath(increase);
        String preorder = "9,3,4,#,#,1,#,#,2,#,6,#,#";
        // threeHundred.isValidSerialization(preorder);
        // threeHundred.coinChange(coins, 11);
        List<String> word1 = Arrays.asList("JFK", "SFO");
        List<String> word2 = Arrays.asList("JFK", "ATL");
        List<String> word3 = Arrays.asList("SFO", "ATL");
        List<String> word4 = Arrays.asList("ATL", "JFK");

        List<String> word5 = Arrays.asList("ATL", "SFO");
        // List<List<String>> airports = Arrays.asList(word1, word2, word3, word4, word5);
        // threeHundred.findItinerary(airports);
        String preoder = "9,3,4,#,#,1,#,#,2,#,6,#,#";
        // threeHundred.isValidSerialization(preoder);
        // String[] palindrome = new String[]{"abcd", "dcba", "lls", "s", "sssll"};
        String[] palindrome = new String[]{"a", "abc", "aba", ""};

        // threeHundred.palindromePairs(palindrome);
        TreeNode root = new TreeNode(3);

        TreeNode two = new TreeNode(2);
        two.right = new TreeNode(3);

        TreeNode three = new TreeNode(3);
        three.right = new TreeNode(1);

        root.left = two;
        root.right = three;

        threeHundred.countBits(2);
    }

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
        if (prerequisites == null || prerequisites.length == 0) {
            return false;
        }
        Map<Integer, List<Integer>> matrix = new HashMap<>();
        int[] degrees = new int[numCourses];

        for (int[] prerequisite : prerequisites) {
            int leftEdge = prerequisite[0];
            int rightEdge = prerequisite[1];

            List<Integer> neighbors = matrix.getOrDefault(rightEdge, new ArrayList<>());

            neighbors.add(leftEdge);

            matrix.put(rightEdge, neighbors);

            degrees[leftEdge]++;
        }
        LinkedList<Integer> linkedList = new LinkedList<>();

        int count = 0;

        for (int i = 0; i < degrees.length; i++) {
            if (degrees[i] == 0) {
                linkedList.offer(i);
            }
        }
        while (!linkedList.isEmpty()) {
            int pop = linkedList.pop();
            count++;

            List<Integer> neighbors = matrix.getOrDefault(pop, new ArrayList<>());

            for (Integer neighbor : neighbors) {
                if (--degrees[neighbor] == 0) {
                    linkedList.offer(neighbor);
                }
            }
        }
        return count == numCourses;
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
        return Math.max(this.intervalRob(nums, 0, nums.length - 2), this.intervalRob(nums, 1, nums.length - 1));
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

    /**
     * 282. Expression Add Operators
     * @param num
     * @param target
     * @return
     */
    public List<String> addOperators(String num, int target) {
        if (num == null || num.isEmpty()) {
            return new ArrayList<>();
        }
        List<String> result = new ArrayList<>();
        internalAddOperators(result, "", "", num, target, 0, 0);
        return result;

    }

    private void internalAddOperators(List<String> result, String s, String operator, String num, int target, long expected, long multi) {
        if (target == expected && num.isEmpty() && !result.contains(s)) {
            result.add(s);
            return;
        }
        if (num.isEmpty()) {
            return;
        }
        for (int i = 1; i <= num.length(); i++) {
            if (i > 1 && num.charAt(0) == '0') {
                continue;
            }
            String prefix = num.substring(0, i);
            long value = Long.parseLong(prefix);
            String remain = num.substring(i);

            internalAddOperators(result, s.isEmpty() ? prefix : s + "+" + prefix, "+", remain, target, operator.isEmpty() ? value : expected + value, value);
            internalAddOperators(result, s.isEmpty() ? prefix : s + "-" + prefix, "-", remain, target, operator.isEmpty() ? value : expected - value, operator.isEmpty() ? value : -value);
            internalAddOperators(result, s.isEmpty() ? prefix : s + "*" + prefix, "*", remain, target, operator.isEmpty() ? value : expected - multi + multi * value, operator.isEmpty() ? value : value * multi);
        }
    }


    private void internalAddOperators(List<String> result, String s, String num, int target, long expected, long multi) {
        if (target == expected && num.isEmpty() && !result.contains(s)) {
            result.add(s);
            return;
        }
        if (num.isEmpty()) {
            return;
        }
        for (int i = 1; i <= num.length(); i++) {
            if (i > 1 && num.charAt(0) == '0') {
                continue;
            }
            String prefix = num.substring(0, i);
            long value = Long.parseLong(prefix);
            String remain = num.substring(i);
            if (s.isEmpty()) {
                internalAddOperators(result, prefix, remain, target, value, value);
            } else {
                internalAddOperators(result, s + "+" + prefix, remain, target, expected + value, value);
                internalAddOperators(result, s + "-" + prefix, remain, target, expected - value, -value);
                internalAddOperators(result, s + "*" + prefix, remain, target, expected - multi + multi * value, multi * value);
            }
        }
    }


    public List<Integer> findMinHeightTrees(int n, int[][] edges) {
        if (n == 1) return Collections.singletonList(0);

        List<Set<Integer>> adj = new ArrayList<>(n);
        for (int i = 0; i < n; ++i) adj.add(new HashSet<>());
        for (int[] edge : edges) {
            adj.get(edge[0]).add(edge[1]);
            adj.get(edge[1]).add(edge[0]);
        }

        List<Integer> leaves = new ArrayList<>();
        for (int i = 0; i < n; ++i)
            if (adj.get(i).size() == 1) leaves.add(i);

        while (n > 2) {
            n -= leaves.size();
            List<Integer> newLeaves = new ArrayList<>();
            for (int i : leaves) {
                int j = adj.get(i).iterator().next();
                adj.get(j).remove(i);
                if (adj.get(j).size() == 1) newLeaves.add(j);
            }
            leaves = newLeaves;
        }
        return leaves;
    }


    /**
     * @param n: An integer
     * @param m: An integer
     * @param operators: an array of point
     * @return: an integer array
     */
    public List<Integer> numIslands2(int n, int m, Point[] operators) {
        if (operators == null || operators.length == 0) {
            return new ArrayList<>();
        }

        List<Integer> result = new ArrayList<>();
        // write your code here
        boolean[][] visited = new boolean[n][m];
        int[] father = constructUnionFather(n, m);
        int[][] matrix = new int[][]{{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        int count = 0;

        for (Point operator : operators) {
            int x = operator.x;
            int y = operator.y;
            if (!visited[x][y]) {
                count++;
                int currentPoint = x * n + y;
                for (int[] direction : matrix) {
                    int edgeRow = x + direction[0];
                    int edgeColumn = y + direction[1];

                    if (inValidEdge(edgeRow, edgeColumn, n, m)) {
                        continue;
                    }
                    if (!visited[edgeRow][edgeColumn]) {
                        continue;
                    }
                    int adjacentPoint = edgeRow * n + edgeColumn;

                    int adjacentFather = findFather(father, adjacentPoint);
                    int currentFather = findFather(father, currentPoint);

                    if (currentFather != adjacentFather) {
                        count--;
                        father[currentFather] = adjacentFather;
                    }
                }
            }
            visited[x][y] = true;
            result.add(count);
        }
        return result;
    }


    private int[] constructUnionFather(int n, int m) {
        int[] father = new int[n * m];
        for (int i = 0; i < father.length; i++) {
            father[i] = i;
        }
        return father;
    }


    private int findFather(int[] father, int x) {
        if (father[x] != x) {
            return father[x] = findFather(father, father[x]);
        }
        return x;
    }

    private boolean inValidEdge(int x, int y, int n, int m) {
        return x < 0 || x >= n || y < 0 || y >= m;
    }


    /**
     * 312. Burst Balloons
     * @param nums
     * @return
     */
    public int maxCoins(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int[] coins = new int[nums.length + 2];

        coins[0] = 1;
        coins[coins.length - 1] = 1;
        System.arraycopy(nums, 0, coins, 1, nums.length);

        int[][] result = new int[coins.length][coins.length];

        for (int k = 2; k < coins.length; k++) {
            for (int left = 0; left < coins.length - k; left++) {
                int right = left + k;
                for (int i = left + 1; i < right; i++) {
                    result[left][right] = Math.max(result[left][right], coins[left] * coins[i] * coins[right] + result[left][i] + result[i][right]);
                }
            }
        }
        return result[0][coins.length - 1];
    }

    /**
     * @param root: the root of tree
     * @return: the vertical order traversal
     */
    public List<List<Integer>> verticalOrder(TreeNode root) {
        // write your code here
        if (root == null) {
            return new ArrayList<>();
        }
        Map<Integer, List<Integer>> results = new TreeMap<Integer, List<Integer>>();
        LinkedList<Integer> linkedList = new LinkedList<>();
        LinkedList<TreeNode> nodeList = new LinkedList<>();
        linkedList.offer(0);
        nodeList.offer(root);

        while (!linkedList.isEmpty()) {
            Integer column = linkedList.poll();

            List<Integer> neighbors = results.getOrDefault(column, new ArrayList<>());

            TreeNode poll = nodeList.poll();

            neighbors.add(poll.val);

            results.put(column, neighbors);

            if (poll.left != null) {
                nodeList.offer(poll.left);
                linkedList.offer(column - 1);
            }
            if (poll.right != null) {
                nodeList.offer(poll.right);
                linkedList.offer(column + 1);
            }

        }
        return new ArrayList<>(results.values());
    }

    private void internalVerticalTree(Map<Integer, List<Integer>> map, TreeNode root, int row, int column) {
        if (root == null) {
            return;
        }
        List<Integer> neighbors = map.getOrDefault(column, new ArrayList<>());
        neighbors.add(root.val);

        map.put(column, neighbors);
        internalVerticalTree(map, root.left, row + 1, column - 1);
        internalVerticalTree(map, root.right, row + 1, column + 1);
    }


    public List<Integer> countSmaller(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new ArrayList<>();
        }
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < nums.length; i++) {
            int count = 0;
            boolean existMin = false;
            for (int j = i + 1; j < nums.length; j++) {
                if (nums[j] < nums[i]) {
                    count++;
                }
            }
            result.add(count);
        }
        return result;
    }

    public String removeDuplicateLetters(String s) {
        if (s == null || s.isEmpty()) {
            return "";
        }
        Map<Character, Integer> map = new TreeMap<>();

        char[] words = s.toCharArray();

        for (char word : words) {
            Integer count = map.getOrDefault(word, 0);

            if (count == 1) {
                continue;
            }
            map.put(word, count + 1);
        }
        StringBuilder builder = new StringBuilder();
        for (Map.Entry<Character, Integer> item : map.entrySet()) {
            Character key = item.getKey();
            builder.append(key);
        }
        return builder.toString();
    }


    public int maxProduct(String[] words) {
        if (words == null || words.length == 0) {
            return 0;
        }
        int[] params = new int[words.length];
        for (int i = 0; i < words.length; i++) {
            String current = words[i];
            for (char currentCharacter : current.toCharArray()) {
                params[i] |= 1 << (currentCharacter - 'a');
            }
        }
        int result = 0;
        for (int i = 0; i < words.length; i++) {
            for (int j = i + 1; j < words.length; j++) {
                if ((params[i] & params[j]) != 0) {
                    continue;
                }
                result = Math.max(result, words[i].length() * words[j].length());
            }
        }
        return result;
    }


    public int bulbSwitch(int n) {
        if (n <= 0) {
            return 0;
        }
        if (n == 1) {
            return 1;
        }
        int count = 0;
        int indexNumber = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < 32; j++) {
                int index = (i % Integer.MAX_VALUE);

                count |= 1 << index;

                indexNumber |= 1 << index;
            }

        }
        for (int j = 2; j <= n; j++) {
            for (int i = 2; i <= 32; i++) {
                if (i % j != 0) {
                    continue;
                }
                if ((indexNumber & (1 << (i - 1))) == 0) {
                    continue;
                }
                count ^= 1 << (i - 1);
            }

        }
        int result = 0;
        for (int i = 0; i < 32; i++) {
            int remainCount = count & (1 << i);
            if (remainCount != 0) {
                result++;
            }
        }
        return result;
    }


    /**
     * @param word: the given word
     * @return: the generalized abbreviations of a word
     *          we will sort your return value in output
     */
    public List<String> generateAbbreviations(String word) {
        // Write your code here
        if (word == null || word.isEmpty()) {
            return new ArrayList<>();
        }
        List<String> result = new ArrayList<>();

        internalGenerate(result, word, 0, "", 0);

        return result;
    }

    private void internalGenerate(List<String> result, String word, int count, String curr, int position) {
        if (position == word.length()) {
            if (count > 0) {
                curr += count;
            }
            result.add(curr);
        } else {
            internalGenerate(result, word, count + 1, curr, position + 1);
            if (count > 0) {
                curr = curr + count + word.charAt(position);
            } else {
                curr += word.charAt(position);
            }
            internalGenerate(result, word, 0, curr, position + 1);
        }
    }


    public int coinChange(int[] coins, int amount) {
        if (coins == null || coins.length == 0) {
            return 0;
        }
        if (amount == 0) {
            return 0;
        }
        Arrays.sort(coins);
        int[] result = new int[amount + 1];
        for (int i = 1; i <= amount; i++) {
            int min = Integer.MAX_VALUE;
            for (int coin : coins) {
                if (i >= coin && result[i - coin] != Integer.MAX_VALUE) {
                    min = Math.min(min, 1 + result[i - coin]);
                }
            }
            result[i] = min;
        }
        return result[amount] == Integer.MAX_VALUE ? -1 : result[amount];
    }


    /**
     * @param n: the number of vertices
     * @param edges: the edges of undirected graph
     * @return: the number of connected components
     */
    public int countComponents(int n, int[][] edges) {
        // write your code here
        if (edges == null) {
            return 0;
        }
        // Arrays.sort(edges, Comparator.comparingInt(o -> o[0]));
        for (int[] edge : edges) {
            int input = edge[0];
            int output = edge[1];
            if (input > output) {
                edge[0] = output;
                edge[1] = input;
            }
        }
        CombinationFind combinationFind = new CombinationFind(n);

        for (int[] edge : edges) {
            int input = edge[0];
            int output = edge[1];
            combinationFind.join(input, output);
        }
        int[] fathers = combinationFind.getFathers();

        Set<Integer> visited = new HashSet<>();

        for (int father : fathers) {
            // visited.add(combinationFind.findFather(father));
            visited.add(father);
        }
        return visited.size();

    }


    public int countComponentsV2(int n, int[][] edges) {
        if (edges == null) {
            return 0;
        }
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int[] edge : edges) {
            int input = edge[0];

            int output = edge[1];

            List<Integer> outputNeighbors = map.getOrDefault(input, new ArrayList<>());

            outputNeighbors.add(output);

            map.put(input, outputNeighbors);

            List<Integer> inputNeighbors = map.getOrDefault(output, new ArrayList<>());
            inputNeighbors.add(input);
            map.put(output, inputNeighbors);
        }
        Set<Integer> visited = new HashSet<>();
        int count = 0;

        for (int i = 0; i < n; i++) {
            if (visited.contains(i)) {
                continue;
            }
            count++;
            internalDFS(visited, map, i);
        }
        return count;
    }

    private void internalDFS(Set<Integer> visited, Map<Integer, List<Integer>> map, int current) {
        if (visited.contains(current)) {
            return;
        }
        visited.add(current);
        List<Integer> neighbors = map.getOrDefault(current, new ArrayList<>());
        for (Integer neighbor : neighbors) {
            if (!visited.contains(neighbor)) {
                internalDFS(visited, map, neighbor);
            }
        }
    }


    /**
     * todo binary indexed tree
     * @param nums: an array
     * @param k: a target value
     * @return: the maximum length of a subarray that sums to k
     */
    public int maxSubArrayLen(int[] nums, int k) {
        // Write your code here
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int[] result = new int[nums.length + 1];
        Map<Integer, Integer> map = new HashMap<>();
        map.put(k, 0);
        int ans = 0;
        for (int i = 1; i <= nums.length; i++) {
            result[i] = result[i - 1] + nums[i - 1];

        }

        return -1;
    }


    public ListNode oddEvenList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode oddNode = new ListNode(0);
        ListNode EvenNode = new ListNode(0);

        ListNode dummyOdd = oddNode;
        ListNode dummyEven = EvenNode;

        int count = 1;
        while (head != null) {
            if (count % 2 == 0) {
                dummyEven.next = head;
                dummyEven = dummyEven.next;
            } else {
                dummyOdd.next = head;
                dummyOdd = dummyOdd.next;
            }
            count++;
            head = head.next;
        }
        dummyOdd.next = EvenNode.next;
        dummyEven.next = null;
        return oddNode.next;
    }

    public int longestIncreasingPath(int[][] matrix) {
        if (matrix == null || matrix.length == 0) {
            return 0;
        }
        int result = 0;
        int[][] visited = new int[matrix.length][matrix[0].length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                int longestLen = internalIncrease(i, j, Integer.MIN_VALUE, matrix, visited);
                result = Math.max(result, longestLen);
            }
        }
        return result;
    }

    private int internalIncrease(int i, int j, int previousValue, int[][] matrix, int[][] visited) {
        if (i < 0 || i == matrix.length || j < 0 || j >= matrix[0].length) {
            return 0;
        }
        if (previousValue >= matrix[i][j]) {
            return 0;
        }
        if (visited[i][j] != 0) {
            return visited[i][j];
        }
        int count = 1;

        count = Math.max(count, 1 + internalIncrease(i - 1, j, matrix[i][j], matrix, visited));

        count = Math.max(count, 1 + internalIncrease(i + 1, j, matrix[i][j], matrix, visited));

        count = Math.max(count, 1 + internalIncrease(i, j - 1, matrix[i][j], matrix, visited));

        count = Math.max(count, 1 + internalIncrease(i, j + 1, matrix[i][j], matrix, visited));

        visited[i][j] = count;

        return count;
    }

    public boolean isValidSerialization(String preorder) {
        if (preorder == null || preorder.isEmpty()) {
            return true;
        }
        String[] nodes = preorder.split(",");
        Stack<String> stack = new Stack<>();

        for (int i = 0; i < nodes.length; i++) {
            String current = nodes[i];

            while (!stack.isEmpty() && current.equals("#") && stack.peek().equals(current)) {
                stack.pop();

                if (stack.isEmpty()) {
                    return false;
                }
                stack.pop();
            }
            stack.push(current);
        }
        return stack.size() == 1 && stack.peek().equals("#");
    }

    public List<String> findItinerary(List<List<String>> tickets) {
        if (tickets == null || tickets.isEmpty()) {
            return new ArrayList<>();
        }
        Map<String, PriorityQueue<String>> map = new HashMap<>();
        for (List<String> ticket : tickets) {
            String departure = ticket.get(0);
            String inbound = ticket.get(1);

            PriorityQueue<String> airport = map.getOrDefault(departure, new PriorityQueue<>());

            airport.add(inbound);
            map.put(departure, airport);
        }
        List<String> result = new ArrayList<>();

        findDFS(result, "JFK", map);

        return result;


    }

    private void findDFS(List<String> result, String departure, Map<String, PriorityQueue<String>> map) {
        PriorityQueue<String> airports = map.getOrDefault(departure, new PriorityQueue<>());
        while (!airports.isEmpty()) {
            String poll = airports.poll();
            findDFS(result, poll, map);
        }
        result.add(departure);
    }

    public void preorderTraverse(TreeNode root) {
        if (root == null) {
            return;
        }
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        while (!stack.isEmpty()) {
            TreeNode pop = stack.pop();

            System.out.println(pop.val);

            if (pop.right != null) {
                stack.push(pop.right);
            }
            if (pop.left != null) {
                stack.push(pop.left);
            }
        }
    }


    public boolean increasingTriplet(int[] nums) {
        if (nums == null || nums.length == 0) {
            return true;
        }
        for (int i = 0; i < nums.length - 1; i++) {
            for (int j = i + 1; j < nums.length; j++) {
                int k = j + 1;
                while (k < nums.length) {
                    if (nums[i] < nums[j] && nums[j] < nums[k]) {
                        return true;
                    }
                    k++;
                }
            }
        }
        return false;

    }

    public List<List<Integer>> palindromePairs(String[] words) {
        if (words == null || words.length == 0) {
            return new ArrayList<>();
        }
        Set<String> visited = new HashSet<>();
        List<List<Integer>> result = new ArrayList<>();
        for (int i = 0; i < words.length; i++) {
            String current = words[i];

            for (int j = 0; j < words.length; j++) {
                if (i == j) {
                    continue;
                }
                String combination = current + words[j];

                if (!visited.contains(combination) || isPalindrome(combination)) {
                    List<Integer> ans = Arrays.asList(i, j);
                    visited.add(combination);
                    result.add(ans);
                }

            }
        }
        return result;
    }


    private boolean isPalindrome(String word) {
        if (word == null || word.length() <= 1) {
            return true;
        }
        Map<Character, Integer> map = new HashMap<>();

        int start = 0;
        int end = word.length() - 1;
        while (start < end) {
            if (word.charAt(start) != word.charAt(end)) {
                return false;
            }
            start++;
            end--;
        }
        return true;
    }


    public int rob(TreeNode root) {
        if (root == null) {
            return 0;
        }
        Map<TreeNode, Integer> map = new HashMap<>();

        return internalRob(map, root);
    }

    private int internalRob(Map<TreeNode, Integer> map, TreeNode root) {
        if (root == null) {
            return 0;
        }
        if (root.left == null && root.right == null) {
            return root.val;
        }
        if (map.containsKey(root)) {
            return map.get(root);
        }
        int value = 0;

        if (root.left != null) {
            value += rob(root.left.left) + rob(root.left.right);
        }
        if (root.right != null) {
            value += rob(root.right.left) + rob(root.right.right);
        }
        int result = Math.max(root.val + value, internalRob(map, root.left) + internalRob(map, root.right));

        map.put(root, result);

        return result;
    }


    public int[] countBits(int n) {
        if (n == 0) {
            return new int[]{};
        }
        int[] result = new int[n + 1];
        for (int i = 0; i <= n; i++) {
            int count = 0;
            for (int j = 0; j < 32; j++) {
                if ((i & (1 << j)) != 0) {
                    count++;
                }
            }
            result[i] = count;
        }
        return result;
    }


}
