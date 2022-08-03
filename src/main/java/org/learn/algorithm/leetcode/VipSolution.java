package org.learn.algorithm.leetcode;

import org.learn.algorithm.datastructure.Interval;
import org.learn.algorithm.datastructure.Point;
import org.learn.algorithm.datastructure.TreeNode;
import org.springframework.web.servlet.mvc.method.annotation.ResponseBodyEmitterReturnValueHandler;

import java.util.*;

/**
 * Vip 题目
 *
 * @author luk
 * @date 2021/4/12
 */
public class VipSolution {

    /**
     * todo
     * 298
     * Binary Tree Longest Consecutive Sequence
     *
     * @param root: the root of binary tree
     * @return: the length of the longest consecutive sequence path
     */
    private int longestV2 = 0;

    public static void main(String[] args) {
        VipSolution solution = new VipSolution();
        int[] nums = new int[]{1, 3, 2};
        solution.isStrobogrammatic("96801866799810896");
//        solution.findMissingRanges(new int[]{2147483647}, 0, 2147483647);
    }

    /**
     * 156 Binary Tree Upside Down
     *
     * @param root: the root of binary tree
     * @return: new root
     */
    public TreeNode upsideDownBinaryTree(TreeNode root) {
        if (root == null || root.left == null) {
            return root;
        }
        TreeNode left = root.left;
        TreeNode upsideDown = upsideDownBinaryTree(left);
        left.left = root.right;
        left.right = root;
        root.left = null;
        root.right = null;
        return upsideDown;
        // write your code here
    }

    /**
     * 161 One Edit Distance
     *
     * @param s: a string
     * @param t: a string
     * @return: true if they are both one edit distance apart or false
     */
    public boolean isOneEditDistance(String s, String t) {
        // write your code here
        if (s == null || t == null) {
            return false;
        }
        if (s.equals(t)) {
            return false;
        }
        int m = s.length();
        int n = t.length();
        int min = Math.min(m, n);
        for (int i = 0; i < min; i++) {
            if (s.charAt(i) != t.charAt(i)) {
                if (m < n) {
                    return s.substring(i).equals(t.substring(i + 1));
                } else if (m == n) {
                    return s.substring(i + 1).equals(t.substring(i + 1));
                } else {
                    return s.substring(i + 1).equals(t.substring(i));
                }
            }
        }
        return false;
    }

    /**
     * todo
     * #163 Missing Ranges
     *
     * @param nums:  a sorted integer array
     * @param lower: An integer
     * @param upper: An integer
     * @return: a list of its missing ranges
     */
    public List<String> findMissingRanges(int[] nums, int lower, int upper) {
        // write your code here
        return internalFindMissingRanges(nums, lower, upper);
    }

    private List<String> internalFindMissingRanges(int[] nums, long lower, long upper) {
        if (nums == null) {
            return new ArrayList<>();
        }
        List<String> result = new ArrayList<>();
        if (nums.length == 0) {
            result.add(range(lower, upper));
            return result;
        }
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] > lower) {
                result.add(range(lower, nums[i] - 1));
            }
            lower = (long) nums[i] + 1;
        }
        if (lower <= upper) {
            result.add(range(lower + 1, upper));
        }
        return result;
    }

    private String range(long lower, long upper) {
        return lower == upper ? String.valueOf(lower) : lower + "->" + upper;
    }

    /**
     * 186 Reverse Words in a String II
     * Medium
     *
     * @param str: a string
     * @return: return a string
     */
    public char[] reverseWords(char[] str) {
        // write your code here
        if (str == null || str.length == 0) {
            return str;
        }
        int startIndex = 0;
        while (startIndex < str.length) {
            int endIndex = startIndex;
            while (endIndex < str.length && str[endIndex] != ' ') {
                endIndex++;
            }
            if (endIndex == str.length || str[endIndex] == ' ') {
                reverseArray(str, startIndex, endIndex);
            }
            startIndex = endIndex + 1;
        }
        reverseArray(str, 0, str.length);
        return str;
    }

    private void reverseArray(char[] str, int start, int end) {
        for (int i = start; i <= (start + end - 1) / 2; i++) {
            swap(str, i, start + end - 1 - i);
        }
    }

    // 单词最短距离

    private void swap(char[] str, int i, int j) {
        char tmp = str[i];
        str[i] = str[j];
        str[j] = tmp;
    }

    // 反转数系列

    /**
     * #243 Shortest Word Distance
     *
     * @param words
     * @param word1
     * @param word2
     * @return
     */
    public int shortestDistance(String[] words, String word1, String word2) {
        int index1 = -1;
        int index2 = -1;
        int result = Integer.MAX_VALUE;
        for (int i = 0; i < words.length; i++) {
            String tmp = words[i];
            if (word1.equals(tmp)) {
                index1 = i;
            } else if (word2.equals(tmp)) {
                index2 = i;
            }
            if (index1 != -1 && index2 != -1) {
                result = Math.min(result, Math.abs(index1 - index2));
            }
        }
        return result;
    }

    /**
     * 246 Strobogrammatic Number
     *
     * @param num
     * @return
     */
    public boolean isStrobogrammatic(String num) {
        if (num == null || num.isEmpty()) {
            return false;
        }
        String reverse = new StringBuilder(num).reverse().toString();

        int len = num.length();

        Map<Character, Character> map = getNum();

        for (int i = 0; i < len; i++) {
            char tmp = num.charAt(i);

            if (map.getOrDefault(tmp, ' ') != reverse.charAt(i)) {
                return false;
            }
        }
        return true;


    }

    private Map<Character, Character> getNum() {
        Map<Character, Character> map = new HashMap<>();
        map.put('0', '0');
        map.put('1', '1');
        map.put('6', '9');
        map.put('8', '8');
        map.put('9', '6');
        return map;
    }

    /**
     * #247 Strobogrammatic Number II
     * Medium
     *
     * @param n: the length of strobogrammatic number
     * @return: All strobogrammatic numbers
     */
    public List<String> findStrobogrammatic(int n) {
        // write your code here
        if (n <= 0) {
            return new ArrayList<>();
        }

        return null;

    }

//    private List<String> generateWords()


    private void intervalFind(List<String> result, String s, int n) {

    }

    /**
     * todo
     * 248 Strobogrammatic Number III
     *
     * @param low
     * @param high
     * @return
     */
    public int strobogrammaticInRange(String low, String high) {
        if (low == null || high == null) {
            return 0;
        }
        int m = low.length();
        int n = high.length();
        int count = 0;
        for (int i = m; i <= n; i++) {
            count += intervalCount("", i, low, high);
            count += intervalCount("0", i, low, high);
            count += intervalCount("1", i, low, high);
            count += intervalCount("8", i, low, high);
        }
        return count;
    }

    private int intervalCount(String path, int len, String low, String high) {
        int count = 0;
        int m = path.length();
        if (m >= len) {
            if (m > len || (len > 1 && path.charAt(0) == '0')) {
                return 0;
            }
            if (m == low.length() && low.compareTo(path) > 0) {
                return 0;
            }
            if (m == high.length() && high.compareTo(path) < 0) {
                return 0;
            }
            count++;
        }
        count += intervalCount("0" + path + "0", len, low, high);
        count += intervalCount("1" + path + "1", len, low, high);
        count += intervalCount("6" + path + "9", len, low, high);
        count += intervalCount("8" + path + "8", len, low, high);
        count += intervalCount("9" + path + "6", len, low, high);

        return count;
    }

    /**
     * 249 Group Shifted Strings
     *
     * @param strings
     * @return
     */
    public List<List<String>> groupStrings(String[] strings) {
        if (strings == null || strings.length == 0) {
            return new ArrayList<>();
        }
        Map<String, List<String>> result = new HashMap<>();
        intervalGroup(result, strings);
        return new ArrayList<>(result.values());
    }

    private void intervalGroup(Map<String, List<String>> result, String[] words) {
        for (String word : words) {
            String shiftWord = shiftWord(word);
            List<String> tmp = result.getOrDefault(shiftWord, new ArrayList<>());
            tmp.add(word);
            result.put(shiftWord, tmp);
        }
    }

    private String shiftWord(String word) {
        StringBuilder buffer = new StringBuilder();
        char[] words = word.toCharArray();
        int dist = word.charAt(0) - 'a';
        for (char c : words) {
            char t = (char) ((c - 'a' - dist + 26) % 26 + 'a');
            buffer.append(t);
        }
        return buffer.toString();
    }

    /**
     * 250
     * Count Univalue Subtrees
     *
     * @param root: the given tree
     * @return: the number of uni-value subtrees.
     */
    public int countUnivalSubtrees(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int count = 0;
        if (isUnivalSubTree(root, root.val)) {
            count++;
        }
        count += countUnivalSubtrees(root.left);
        count += countUnivalSubtrees(root.right);
        return count;
    }

    private boolean isUnivalSubTree(TreeNode root, int val) {
        if (root == null) {
            return true;
        }
        return root.val == val && isUnivalSubTree(root.left, root.val) && isUnivalSubTree(root.right, root.val);
    }

    /**
     * 252
     * Meeting Rooms
     *
     * @param intervals: an array of meeting time intervals
     * @return: if a person could attend all meetings
     */
    public boolean canAttendMeetings(List<Interval> intervals) {
        // Write your code here
        if (intervals == null || intervals.isEmpty()) {
            return false;
        }
        intervals.sort(Comparator.comparingInt(o -> o.start));
        Interval pre = intervals.get(0);
        for (int i = 1; i < intervals.size(); i++) {
            Interval current = intervals.get(i);
            if (current.start < pre.end) {
                return false;
            }
            pre = current;
        }
        return true;
    }

    /**
     * todo
     * 253
     * Meeting Rooms II
     *
     * @param intervals: an array of meeting time intervals
     * @return: the minimum number of conference rooms required
     */
    public int minMeetingRooms(List<Interval> intervals) {
        // Write your code here
        if (intervals == null || intervals.isEmpty()) {
            return 0;
        }
        PriorityQueue<Integer> queue = new PriorityQueue<>();
        intervals.sort(Comparator.comparingInt(o -> o.start));
        for (Interval interval : intervals) {
            if (!queue.isEmpty() && queue.peek() < interval.start) {
                queue.poll();
            }
            queue.offer(interval.end);
        }
        return queue.size();
    }

    /**
     * 255
     * Verify Preorder Sequence in Binary Search Tree
     *
     * @param preorder: List[int]
     * @return: return a boolean
     */
    public boolean verifyPreorder(int[] preorder) {
        if (preorder == null || preorder.length == 0) {
            return false;
        }
        return intervalVerifyPreorder(Integer.MIN_VALUE, Integer.MAX_VALUE, 0, preorder.length - 1, preorder);
    }

    private boolean intervalVerifyPreorder(int minValue, int maxValue, int start, int end, int[] preorder) {
        if (start > end) {
            return true;
        }
        int val = preorder[start], i = 0;
        if (val <= minValue || val >= maxValue) {
            return false;
        }
        for (i = start + 1; i <= end; ++i) {
            if (preorder[i] >= val) {
                break;
            }
        }
        return intervalVerifyPreorder(minValue, val, start + 1, i - 1, preorder) &&
                intervalVerifyPreorder(val, maxValue, i, end, preorder);
    }

    /**
     * todo
     * #259 3Sum Smaller
     * Medium
     *
     * @param nums:   an array of n integers
     * @param target: a target
     * @return: the number of index triplets satisfy the condition nums[i] + nums[j] + nums[k] < target
     */
    public int threeSumSmaller(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        Arrays.sort(nums);
        int count = 0;
        for (int i = 0; i < nums.length - 2; i++) {
            int start = i + 1;
            int end = nums.length - 1;
            while (start < end) {
                int val = nums[i] + nums[start] + nums[end];
                if (val < target) {
                    count += end - start;
                    start++;
                } else {
                    end--;
                }
            }
        }
        return count;
        // Write your code here
    }

    /**
     * 270
     * Closest Binary Search Tree Value
     *
     * @param root:   the given BST
     * @param target: the given target
     * @return: the value in the BST that is closest to the target
     */
    public int closestValue(TreeNode root, double target) {
        // write your code here
        int result = Integer.MAX_VALUE;
        while (root != null) {
            if (result == Integer.MAX_VALUE) {
                result = root.val;
            } else {
                double diff = Math.abs(root.val - target);
                double last = Math.abs(result - target);
                if (last > diff) {
                    result = root.val;
                }
                if (root.val < target) {
                    root = root.right;
                } else {
                    root = root.left;
                }
            }
        }
        return result;
    }


    /**
     * 270. Closest Binary Search Tree Value
     *
     * @param root
     * @param target
     * @param k
     * @return
     */
    public List<Integer> closestKValues(TreeNode root, double target, int k) {
        if (root == null) {
            return new ArrayList<>();
        }
        PriorityQueue<Integer> queue = new PriorityQueue<>();
        Stack<TreeNode> stack = new Stack<>();
        TreeNode p = root;
        while (!stack.isEmpty() || p != null) {
            while (p != null) {
                stack.push(p);
                p = p.left;
            }
            p = stack.pop();
            if (queue.size() < k) {
                queue.offer(p.val);
            } else if (Math.abs(queue.peek() - target) > Math.abs(p.val - target)) {
                queue.poll();
                queue.offer(p.val);
            } else {
                break;
            }
            p = p.right;
        }
        return new ArrayList<>(queue);
    }


    /**
     * todo
     * 276. Paint Fence
     *
     * @param n: non-negative integer, n posts
     * @param k: non-negative integer, k colors
     * @return: an integer, the total number of ways
     */
    public int numWays(int n, int k) {
        // write your code here
        if (n <= 0) {
            return 0;
        }
        int same = 0;
        int diff = k;
        for (int i = 2; i <= n; i++) {
            int pre = diff;
            diff = (same + diff) * (k - 1);
            same = pre;
        }
        return same + diff;
    }


    /**
     * @param nums: A list of integers
     * @return: nothing
     */
    public void wiggleSort(int[] nums) {
        // write your code here
        if (nums == null || nums.length <= 1) {
            return;
        }
        Arrays.sort(nums);
        for (int i = 1; i < nums.length; i = i + 2) {
            if (nums[i] > nums[i - 1]) {
                swap(nums, i, i - 1);
            }
        }
    }

    public void wiggleSortV2(int[] nums) {
        if (nums == null || nums.length <= 1) {
            return;
        }
        for (int i = 1; i < nums.length; i++) {
            boolean odd = i % 2 == 1;
            boolean errorFormat = false;
            if (odd && nums[i] < nums[i - 1]) {
                errorFormat = true;
            }
            if (!odd && nums[i] > nums[i - 1]) {
                errorFormat = true;
            }
            if (errorFormat) {
                swap(nums, i, i - 1);
            }

        }
    }

    private void swap(int[] nums, int i, int j) {
        int val = nums[i];
        nums[i] = nums[j];
        nums[j] = val;
    }

    /**
     * todo
     * 291
     * Word Pattern II
     *
     * @param pattern: a string,denote pattern string
     * @param str:     a string, denote matching string
     * @return: a boolean
     */
    public boolean wordPatternMatch(String pattern, String str) {
        if (pattern == null || str == null) {
            return false;
        }
        Map<Character, String> map = new HashMap<>();
        Set<String> used = new HashSet<>();
        return intervalWordPattern(map, used, pattern, str);
        // write your code here
    }

    private boolean intervalWordPattern(Map<Character, String> map, Set<String> used, String pattern, String str) {
        if (pattern.isEmpty()) {
            return str.isEmpty();
        }
        char charAt = pattern.charAt(0);
        int len = str.length();
        if (map.containsKey(charAt)) {
            String tmp = map.get(charAt);
            if (!str.startsWith(tmp)) {
                return false;
            }
            return intervalWordPattern(map, used, pattern.substring(1), str.substring(tmp.length()));
        }
        for (int i = 0; i < len; i++) {
            String tmp = str.substring(0, i + 1);
            if (used.contains(tmp)) {
                continue;
            }
            map.put(charAt, tmp);
            used.add(tmp);
            if (intervalWordPattern(map, used, pattern.substring(1), str.substring(i + 1))) {
                return true;
            }
            map.remove(charAt);
            used.remove(tmp);
        }
        return false;
    }

    /**
     * 293
     * Flip Game
     *
     * @param s: the given string
     * @return: all the possible states of the string after one valid move
     */
    public List<String> generatePossibleNextMoves(String s) {
        List<String> result = new ArrayList<>();
        if (s == null) {
            return result;
        }
        int len = s.length();
        if (len <= 1) {
            return result;
        }
        int endIndex = 0;
        while (endIndex < len) {
            int index = s.indexOf("++", endIndex);
            if (index == -1) {
                break;
            }
            String tmp = s.substring(0, index) + "--" + s.substring(index + 2);
            result.add(tmp);
            endIndex = index + 1;
        }
        return result;
        // write your code here
    }

    /**
     * 294
     * Flip Game II
     *
     * @param s: the given string
     * @return: if the starting player can guarantee a win
     */
    public boolean canWin(String s) {
        // write your code here
        if (s == null) {
            return false;
        }
        int len = s.length();
        if (len < 2) {
            return false;
        }
        for (int i = 0; i < len; i++) {
            if (s.startsWith("++", i)) {
                String t = s.substring(0, i) + "--" + s.substring(i + 2);
                if (!canWin(t)) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * 曼哈顿距离法
     * 296
     * Best Meeting Point
     *
     * @param grid: a 2D grid
     * @return: the minimize travel distance
     */
    public int minTotalDistance(int[][] grid) {
        // Write your code here
        if (grid == null || grid.length == 0) {
            return 0;
        }
        int row = grid.length;
        int column = grid[0].length;
        List<Integer> pointRow = new ArrayList<>();
        List<Integer> pointColumn = new ArrayList<>();
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (grid[i][j] == 1) {
                    pointRow.add(i);
                    pointColumn.add(j);

                }
            }
        }
        return getDistance(pointRow) + getDistance(pointColumn);
    }

    private int getDistance(List<Integer> tmp) {
        tmp.sort(Comparator.comparingInt(o -> o));
        int start = 0;
        int end = tmp.size() - 1;
        int distance = 0;
        while (start < end) {
            distance += tmp.get(start) + tmp.get(end);

            start++;
            end--;
        }
        return distance;
    }

    public int longestConsecutive2(TreeNode root) {
        if (root == null) {
            return 0;
        }
        intervalConsecutive(root, null, root.right);
        return longestV2;
        // write your code here
    }

    private int intervalConsecutive(TreeNode root, TreeNode prev, TreeNode reverse) {
        if (root == null) {
            return 0;
        }
        TreeNode reverseRight = null;
        TreeNode reverseLeft = null;
        if (prev != null) {
            reverseRight = prev.right;
            reverseLeft = prev.left;
        }
        int left = intervalConsecutive(root.left, root, reverseRight);

        int right = intervalConsecutive(root.right, root, reverseLeft);

        if (reverse != null && reverse.val != root.val) {
            longestV2 = Math.max(longestV2, left + right + 1);
        } else {
            longestV2 = Math.max(longestV2, Math.max(left, right) + 1);
        }

        return Math.max(left, right) + 1;
    }

    private int intervalConsecutive(TreeNode root, TreeNode prev, int count) {
        if (root == null) {
            return count;
        }
        if (prev != null && Math.abs(prev.val - root.val) != 1) {
            return intervalConsecutive(root, null, 1);
        }
        TreeNode preLeft = prev == null ? null : prev.left;
        TreeNode preRight = prev == null ? null : prev.right;
        if (preLeft != null && preRight != null && preLeft.val == preRight.val) {
            return Math.max(intervalConsecutive(root.left, root, count + 1), intervalConsecutive(root.left, root, count + 1));
        }
        return 0;
    }


    /**
     * 302
     * Smallest Rectangle Enclosing Black Pixels
     *
     * @param image: a binary matrix with '0' and '1'
     * @param x:     the location of one of the black pixels
     * @param y:     the location of one of the black pixels
     * @return: an integer
     */
    public int minArea(char[][] image, int x, int y) {
        // write your code here
        int row = image.length - 1;

        int column = image[0].length - 1;

        int left = getAreaEdge(image, 0, row, 0, y, true);
        int right = getAreaEdge(image, 0, row, y, column, true);
        int top = getAreaEdge(image, 0, column, 0, x, false);
        int bottom = getAreaEdge(image, 0, column, x + 1, row, true);
        return (right - left) * (bottom - top);
    }

    private int getAreaEdge(char[][] image, int i, int row, int y, int column, boolean vertical) {
        return 0;
    }


    /**
     * todo 并查集
     * 305
     * Number of Islands II
     *
     * @param n:         An integer
     * @param m:         An integer
     * @param operators: an array of point
     * @return: an integer array
     */
    public List<Integer> numIslands2(int n, int m, Point[] operators) {
        // write your code here
        int[][] matrix = new int[n][m];
        List<Integer> result = new ArrayList<>();
        PriorityQueue<Point> queue = new PriorityQueue<>(new Comparator<Point>() {
            @Override
            public int compare(Point o1, Point o2) {
                if (o1.x == o2.x) {
                    return o1.y - o2.y;
                }
                return o1.x - o2.x;
            }
        });
        for (Point operator : operators) {
            if (queue.isEmpty()) {
                queue.offer(operator);
            } else {


            }
            result.add(queue.size());
        }
        return result;
    }


    /**
     * 311.
     * 稀疏矩阵乘法
     *
     * @param A: a sparse matrix
     * @param B: a sparse matrix
     * @return: the result of A * B
     */
    public int[][] multiply(int[][] A, int[][] B) {
        // write your code here
        return null;
    }


    /**
     * 314
     * Binary Tree Vertical Order Traversal
     *
     * @param root
     * @return
     */
    public List<List<Integer>> verticalOrder(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        TreeMap<Integer, List<Integer>> map = new TreeMap<>();
        LinkedList<TreeNode> linkedList = new LinkedList<>();
        Map<TreeNode, Integer> heightMap = new HashMap<>();
        linkedList.offer(root);
        while (!linkedList.isEmpty()) {
            TreeNode poll = linkedList.poll();
            Integer height = heightMap.getOrDefault(poll, 0);
            heightMap.put(poll, height);
            List<Integer> tmp = map.getOrDefault(height, new ArrayList<>());
            tmp.add(poll.val);
            if (poll.left != null) {
                linkedList.offer(poll.left);
                heightMap.put(poll.left, height - 1);
            }
            if (poll.right != null) {
                linkedList.offer(poll.right);
                heightMap.put(poll.right, height + 1);
            }
            map.put(height, tmp);
        }
        return new ArrayList<>(map.values());
    }

    /**
     * 314. Binary Tree Vertical Order Traversal
     *
     * @param root
     * @return
     */
    public List<List<Integer>> verticalOrderV2(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        LinkedList<TreeNode> linkedList = new LinkedList<>();
        linkedList.offer(root);
        TreeMap<Integer, List<Integer>> map = new TreeMap<>();
        LinkedList<Integer> heightList = new LinkedList<>();
        heightList.offer(1);
        while (!linkedList.isEmpty()) {
            TreeNode poll = linkedList.poll();
            int height = heightList.poll();
            List<Integer> tmp = map.getOrDefault(height, new ArrayList<>());
            tmp.add(poll.val);
            map.put(height, tmp);
            if (poll.left != null) {
                heightList.offer(height - 1);
                linkedList.offer(poll.left);
            }
            if (poll.right != null) {
                heightList.offer(height + 1);
                linkedList.offer(poll.right);
            }
        }
        return new ArrayList<>(map.values());
    }


}
