package org.learn.algorithm.leetcode;

import org.learn.algorithm.datastructure.Interval;
import org.learn.algorithm.datastructure.TreeNode;
import sun.jvm.hotspot.ui.tree.RevPtrsTreeNodeAdapter;

import java.util.*;

/**
 * Vip 题目
 *
 * @author luk
 * @date 2021/4/12
 */
public class VipSolution {

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
        if (start <= 0 || end >= preorder.length) {
            return true;
        }
        if (preorder[start] < minValue || preorder[end] > maxValue) {
            return false;
        }
        int index = start;
        for (int i = start; i <= end; i++) {
            if (preorder[i] > preorder[start]) {
                index = i;
                break;
            }
        }
        return intervalVerifyPreorder(minValue, start, index - 1, preorder, preorder[index]) &&
                intervalVerifyPreorder(preorder[index], index + 1, end, preorder, maxValue);
    }


    public static void main(String[] args) {
        VipSolution solution = new VipSolution();
        String s = "the sky is blue";
        List<Interval> list = new ArrayList<>();
        list.add(new Interval(0, 30));
        list.add(new Interval(5, 10));
        list.add(new Interval(15, 20));
        solution.minMeetingRooms(list);

    }


}
