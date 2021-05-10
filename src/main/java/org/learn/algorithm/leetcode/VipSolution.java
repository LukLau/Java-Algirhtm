package org.learn.algorithm.leetcode;

import org.learn.algorithm.datastructure.Interval;
import org.learn.algorithm.datastructure.Point;
import org.learn.algorithm.datastructure.TreeNode;

import java.util.*;

/**
 * Vip 题目
 *
 * @author luk
 * @date 2021/4/12
 */
public class VipSolution {

    public static void main(String[] args) {
        VipSolution solution = new VipSolution();
        String s = "++++";
        TreeNode root = new TreeNode(1);

        TreeNode left = new TreeNode(2);

        left.left = new TreeNode(3);

        TreeNode right = new TreeNode(0);
        root.left = left;
        root.right = right;

        TreeNode r1 = new TreeNode(3);
        r1.left = new TreeNode(2);
        r1.right = new TreeNode(2);


        solution.longestConsecutive2(r1);
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
        TreeNode node = upsideDownBinaryTree(root.left);

        root.left.left = root.right;

        root.left.right = root;

        root.left = null;

        root.right = null;

        return node;
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
        if (s == null || t == null) {
            return false;
        }
        if (s.equals(t)) {
            return false;
        }
        int m = s.length();
        int n = t.length();
        int len = Math.min(m, n);
        for (int i = 0; i < len; i++) {
            if (s.charAt(i) != t.charAt(i)) {
                if (m == n) {
                    return s.substring(i + 1).equals(t.substring(i + 1));
                } else if (m < n) {
                    return s.substring(i).equals(t.substring(i + 1));
                } else {
                    return s.substring(i + 1).equals(t.substring(i));
                }
            }
        }
        return Math.abs(m - n) <= 1;

        // write your code here
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
        if (nums == null) {
            return new ArrayList<>();
        }
        List<String> result = new ArrayList<>();
        for (int num : nums) {
            if (num > lower && num >= lower + 1) {
                String tmp = range(lower, num - 1);
                result.add(tmp);
            }
            if (num == upper) {
                return result;
            }
            lower = num + 1;
        }
        if (lower <= upper) {
            result.add(range(lower, upper));
        }
        return result;
    }

    private String range(int lower, int upper) {
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
            return new char[]{};
        }
        int endIndex = 0;
        while (endIndex < str.length) {
            int startIndex = endIndex;
            while (endIndex < str.length && str[endIndex] != ' ') {
                endIndex++;
            }
            if (endIndex == str.length || str[endIndex] == ' ') {
                reverseArray(str, startIndex, endIndex);
            }
            endIndex++;
        }
        reverseArray(str, 0, str.length);
        return str;
    }

    private void reverseArray(char[] str, int start, int end) {
        for (int i = start; i <= (start + end - 1) / 2; i++) {
            swap(str, i, start + end - 1 - i);
        }
    }

    private void swap(char[] str, int i, int j) {
        char tmp = str[i];
        str[i] = str[j];
        str[j] = tmp;
    }

    // 单词最短距离


    /**
     * #243 Shortest Word Distance
     *
     * @param words
     * @param word1
     * @param word2
     * @return
     */
    public int shortestDistance(String[] words, String word1, String word2) {
        // Write your code here
        int index1 = -1;
        int index2 = -1;
        int result = Integer.MAX_VALUE;
        for (int i = 0; i < words.length; i++) {
            String word = words[i];
            if (word.equals(word1)) {
                index1 = i;
            } else if (word.equals(word2)) {
                index2 = i;
            }
            if (index1 != -1 && index2 != -1) {
                result = Math.min(result, Math.abs(index1 - index2));
            }
        }
        return result;
    }

    // 反转数系列


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
        Map<Character, Character> map = getNum();
        int len = num.length();
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < len; i++) {
            char tmp = num.charAt(i);
            Character reverse = map.get(tmp);
            if (reverse == null) {
                return false;
            }
            builder.append(reverse);
        }
        return num.equals(builder.reverse().toString());
    }

    private Map<Character, Character> getNum() {
        Map<Character, Character> map = new HashMap<>();
        map.put('1', '1');
        map.put('6', '9');
        map.put('8', '8');
        map.put('9', '6');
        map.put('0', '0');
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
        if (n < 0) {
            return new ArrayList<>();
        }
        if (n == 1) {
            return Arrays.asList("0", "1", "8");
        }
        List<String> result = new ArrayList<>();
        intervalFind(result, "", n);
        intervalFind(result, "1", n);
        intervalFind(result, "8", n);
        intervalFind(result, "0", n);
        return result;
    }

    private void intervalFind(List<String> result, String s, int n) {
        if (s.length() == n) {
            result.add(s);
            return;
        }
        if (s.length() > n - 2) {
            return;
        }

        if (s.length() != n - 2) {
            intervalFind(result, "0" + s + "0", n);
        }
        intervalFind(result, "1" + s + "1", n);
        intervalFind(result, "6" + s + "9", n);
        intervalFind(result, "8" + s + "8", n);
        intervalFind(result, "9" + s + "6", n);
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
            count += countStrobogrammatic("", i, low, high);
            count += countStrobogrammatic("0", i, low, high);
            count += countStrobogrammatic("1", i, low, high);
            count += countStrobogrammatic("8", i, low, high);
        }
        return count;
    }

    private int countStrobogrammatic(String path, int len, String low, String high) {
        int count = 0;
        int m = path.length();
        if (m >= len) {
            if (m > len || (len > 1 && path.charAt(0) == '0')) {
                return 0;
            }
            if (len == low.length() && path.compareTo(low) < 0) {
                return 0;
            }
            if (len == high.length() && path.compareTo(high) > 0) {
                return 0;
            }
            count++;
        }
        count += countStrobogrammatic("0" + path + "0", len, low, high);
        count += countStrobogrammatic("1" + path + "1", len, low, high);
        count += countStrobogrammatic("6" + path + "9", len, low, high);
        count += countStrobogrammatic("8" + path + "8", len, low, high);
        count += countStrobogrammatic("9" + path + "6", len, low, high);
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
        Map<String, List<String>> map = new HashMap<>();
        for (String word : strings) {
            String shiftStr = shiftStr(word);
            List<String> tmp = map.getOrDefault(shiftStr, new ArrayList<>());
            tmp.add(word);
            map.put(shiftStr, tmp);
        }
        return new ArrayList<>(map.values());
    }

    private String shiftStr(String str) {
        StringBuilder buffer = new StringBuilder();
        char[] words = str.toCharArray();
        int dist = str.charAt(0) - 'a';
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
        // write your code here
        return count;
    }

    private boolean isUnivalSubTree(TreeNode root, int val) {
        if (root == null) {
            return true;
        }
        return root.val == val && isUnivalSubTree(root.left, root.val) && isUnivalSubTree(root.right, val);
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
            return true;
        }
        intervals.sort(Comparator.comparingInt(o -> o.start));
        int len = intervals.size();
        for (int i = 1; i < len; i++) {
            Interval current = intervals.get(i);
            Interval pre = intervals.get(i - 1);
            if (current.start < pre.end) {
                return false;
            }
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
        if (intervals == null || intervals.isEmpty()) {
            return 0;
        }
        intervals.sort(Comparator.comparingInt(o -> o.start));
        PriorityQueue<Interval> queue = new PriorityQueue<>(Comparator.comparing(item -> item.end));
        for (Interval interval : intervals) {
            if (!queue.isEmpty() && interval.start >= queue.peek().end) {
                queue.poll();
            }
            queue.offer(interval);
        }
        return queue.size();
        // Write your code here
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
            return true;
        }
        return intervalVerifyPreorder(Integer.MIN_VALUE, 0, preorder.length - 1, preorder, Integer.MAX_VALUE);
        // write your code here
    }

    private boolean intervalVerifyPreorder(int minValue, int start, int end, int[] preorder, int maxValue) {
        if (start == end) {
            return true;
        }
        int val = preorder[start];
        if (val <= minValue || val >= maxValue) {
            return false;
        }
        int i = 0;
        for (i = start + 1; i <= end; ++i) {
            if (preorder[i] >= val) {
                break;
            }
        }
        return intervalVerifyPreorder(minValue, start, i - 1, preorder, val)
                && intervalVerifyPreorder(val, i, end, preorder, maxValue);

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
        int result = Integer.MIN_VALUE;
        while (root != null) {
            if (result == Integer.MIN_VALUE) {
                result = root.val;
            } else {
                double diff = Math.abs(root.val - target);
                double last = Math.abs(result - target);
                if (last < diff) {
                    result = root.val;
                }
            }
            if (root.val < target) {
                root = root.right;
            } else {
                root = root.left;
            }
        }
        return result;
        // write your code here
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


    /**
     * todo
     * 298
     * Binary Tree Longest Consecutive Sequence
     *
     * @param root: the root of binary tree
     * @return: the length of the longest consecutive sequence path
     */
    private int longestV2 = 0;

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


}
