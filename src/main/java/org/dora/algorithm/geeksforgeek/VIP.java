package org.dora.algorithm.geeksforgeek;

import org.dora.algorithm.datastructe.Interval;
import org.dora.algorithm.datastructe.TreeNode;

import java.util.*;

/**
 * @author dora
 * @date 2019/11/29
 */
public class VIP {


    /**
     * 250. Count Univalue Subtrees
     *
     * @param root
     * @return
     */
    int subCount = 0;

    public static void main(String[] args) {
        VIP vip = new VIP();
        List<Interval> list = new ArrayList<>();
        Interval one = new Interval(0, 30);
        Interval two = new Interval(5, 10);
        Interval three = new Interval(15, 20);
        list.add(one);
        list.add(two);
        list.add(three);
        vip.minMeetingRooms(list);
    }

    /**
     * 151 Binary Tree Upside Down Medium
     */
    public TreeNode upsizeDownBinaryTree(TreeNode root) {
        if (root == null || root.left == null) {
            return root;
        }
        TreeNode left = root.left;
        TreeNode right = root.right;
        TreeNode ans = this.upsizeDownBinaryTree(left);
        left.left = right;
        left.right = root;
        root.left = null;
        root.right = null;
        return ans;

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
        int m = s.length();

        int n = t.length();

        if (m > n) {
            return isOneEditDistance(t, s);
        }
        int diff = Math.abs(m - n);
        if (diff > 1) {
            return false;
        }
        for (int i = 0; i < m; i++) {
            if (s.charAt(i) != t.charAt(i)) {
                if (diff == 0) {
                    return s.charAt(i + 1) == t.charAt(i + 1);
                } else {
                    return s.charAt(i) == t.charAt(i + 1);
                }
            }
        }
        return m != n;
    }

    /**
     * todo
     * 163 Missing Ranges
     *
     * @param nums:  a sorted integer array
     * @param lower: An integer
     * @param upper: An integer
     * @return: a list of its missing ranges
     */
    public List<String> findMissingRanges(int[] nums, int lower, int upper) {
        List<String> ans = new ArrayList<>();
        if (nums == null || nums.length == 0) {
            ans.add(getRange(lower, upper));
            return ans;
        }
        if (nums[0] > lower) {
            ans.add(getRange(lower, nums[0] - 1));
        }
        for (int i = 1; i < nums.length; i++) {
            int diff = nums[i] - nums[i - 1];
            if (diff > 1) {
                ans.add(getRange(nums[i - 1] + 1, nums[i] - 1));
            }
        }
        if (nums[nums.length - 1] < upper) {
            ans.add(getRange(nums[nums.length - 1] + 1, upper));
        }
        return ans;
    }

    private String getRange(int start, int end) {
        if (start == end) {
            return String.valueOf(start);
        }
        return start + "->" + end;
    }

    public List<String> findMissingRangesV2(int[] nums, int lower, int upper) {
        List<String> ans = new ArrayList<>();
        if (nums == null || nums.length == 0) {
            ans.add(getRange(lower, upper));
            return ans;
        }
        if (nums[0] > lower) {
            ans.add(getRange(lower, nums[0] - 1));
        }
        int prev = nums[0];
        for (int i = 1; i <= nums.length; i++) {
            int cur = i == nums.length ? upper + 1 : nums[i];
            long diff = (long) cur - (long) prev;
            if (diff > 1) {
                ans.add(getRange(prev + 1, cur - 1));
            }
            prev = cur;
        }
        return ans;
    }

    /**
     * 解题思路 单纯构造一个hash表
     * 247 Strobogrammatic Number II
     *
     * @param num
     * @return
     */
    public boolean isStrobogrammatic(String num) {
        // write your code here
        if (num == null || num.length() == 0) {
            return false;
        }
        HashMap<Character, Character> map = new HashMap<>();
        map.put('0', '0');
        map.put('1', '1');
        map.put('6', '9');
        map.put('8', '8');
        map.put('9', '6');
        for (int i = 0; i <= num.length() / 2; i++) {
            Character character = map.getOrDefault(num.charAt(i), ' ');
            if (character.equals(num.charAt(num.length() - 1 - i))) {
                return false;
            }
        }
        return true;
    }

    /**
     * 247 Strobogrammatic Number II
     *
     * @param n: the length of strobogrammatic number
     * @return: All strobogrammatic numbers
     */
    public List<String> findStrobogrammatic(int n) {
        // write your code here
        if (n <= 0) {
            return new ArrayList<>();
        }
        return intervalStrobogrammatic(n, n);

    }

    private List<String> intervalStrobogrammatic(int n, int m) {
        if (n == 0) {
            return Arrays.asList("");
        }
        if (n == 1) {
            return Arrays.asList("0", "1", "8");
        }
        List<String> list = this.intervalStrobogrammatic(n - 2, m);
        List<String> result = new ArrayList<>();
        for (int i = 0; i < list.size(); i++) {
            String item = list.get(i);

            if (n != m) {
                result.add("0" + item + "0");
            }

            result.add("1" + item + "1");
            result.add("6" + item + "9");
            result.add("8" + item + "8");
            result.add("9" + item + "6");
        }
        return result;
    }

    /**
     * 247 Strobogrammatic Number II
     *
     * @param n
     * @return
     */
    public List<String> findStrobogrammaticV2(int n) {
        List<String> result = new ArrayList<>();
        if (n == 0) {
            return new ArrayList<>();
        }
        LinkedList<String> queue = new LinkedList<>();

        //base case for even number
        String[] candidates = new String[]{"11", "69", "96", "88", "00"};

        String[] oddBaseCase = new String[]{"0", "1", "8"};

        List<String> list = n % 2 == 0 ? Arrays.asList(candidates) : Arrays.asList(oddBaseCase);
        queue.addAll(list);
        while (!queue.isEmpty()) {
            String s = queue.poll();
            if (s.length() == n && (s.length() == 1 || s.charAt(0) != '0')) {
                result.add(s);
            }
            if (s.length() > n) {
                return result;
            }
            for (int i = 0; i < candidates.length; i++) {
                queue.offer(candidates[i].charAt(0) + s + candidates[i].charAt(1));
            }
        }
        return result;
    }

    /**
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
        int[] result = new int[]{0};
        for (int i = low.length(); i <= high.length(); i++) {
            compute(i, "", low, high, result);
            compute(i, "0", low, high, result);
            compute(i, "1", low, high, result);
            compute(i, "8", low, high, result);
        }
        return result[0];
    }

    private void compute(int len, String item, String low, String high, int[] result) {
        if (item.length() >= len) {
            if (item.length() > len && (len != -1 && item.charAt(0) == '0')) {
                return;
            }
            if ((item.length() == low.length() && item.compareTo(low) < 0) || (item.length() == high.length() && item.compareTo(high) > 0)) {
                return;
            }
            int count = result[0];
            result[0] = ++count;
        }
        compute(len, "0" + item + "0", low, high, result);
        compute(len, "1" + item + "1", low, high, result);
        compute(len, "6" + item + "9", low, high, result);
        compute(len, "8" + item + "8", low, high, result);
        compute(len, "9" + item + "6", low, high, result);
    }

    /**
     * 248 Strobogrammatic Number III
     *
     * @param low
     * @param high
     * @return
     */
    public int strobogrammaticInRangeV2(String low, String high) {
        int[] result = new int[]{0};
        intervalCountInRange("", low, high, result);
        intervalCountInRange("0", low, high, result);
        intervalCountInRange("1", low, high, result);
        intervalCountInRange("8", low, high, result);
        return result[0];
    }

    private void intervalCountInRange(String item, String low, String high, int[] result) {
        int len = item.length();
        boolean correctRange = len <= low.length() && len <= high.length();

        if (correctRange) {
            int count = result[0];
            if (len == high.length() && item.compareTo(high) > 0) {
                return;
            }
            if (!(len > 1 && item.charAt(0) == '0') && !(len == low.length() && item.compareTo(low) > 0)) {
                ++count;
                result[0] = count;
            }

        }
        if (len + 2 > high.length()) {
            return;
        }
        this.intervalCountInRange("0" + item + "0", low, high, result);
        this.intervalCountInRange("1" + item + "1", low, high, result);
        this.intervalCountInRange("6" + item + "9", low, high, result);
        this.intervalCountInRange("8" + item + "8", low, high, result);
        this.intervalCountInRange("9" + item + "6", low, high, result);
    }

    /**
     * 249. Group Shifted Strings
     *
     * @param strings
     * @return
     */
    List<List<String>> groupStrings(List<String> strings) {
        return null;
    }

    /**
     * 250. Count Univalue Subtrees
     *
     * @param root
     * @return
     */
    public int countUnivalSubtrees(TreeNode root) {
        if (root == null) {
            return 0;
        }
        if (this.intervalSubTree(root, root.val)) {
            subCount++;
        }
        countUnivalSubtrees(root.left);
        countUnivalSubtrees(root.right);
        return subCount;
    }

    private boolean intervalSubTree(TreeNode root, int val) {
        if (root == null) {
            return true;
        }
        if (root.val == val) {
            return intervalSubTree(root.left, val) && intervalSubTree(root.right, val);
        }
        return false;
    }

    /**
     * @param root
     * @return
     */
    public int countUnivalSubtreesV2(TreeNode root) {
        if (root == null) {
            return 0;
        }
        intervalSubTreeV2(root, -1);
        return subCount;
    }

    private boolean intervalSubTreeV2(TreeNode root, int val) {
        if (root == null) {
            return true;
        }
        if (!(intervalSubTree(root.left, root.val) && intervalSubTree(root.right, root.val))) {
            return false;
        }
        subCount++;
        return root.val == val;
    }


    /**
     * 252 Meeting Rooms
     *
     * @param intervals: an array of meeting time intervals
     * @return: if a person could attend all meetings
     */
    public boolean canAttendMeetings(List<Interval> intervals) {
        if (intervals == null || intervals.isEmpty()) {
            return true;
        }
        Interval[] array = intervals.toArray(new Interval[]{});
        Arrays.sort(array, Comparator.comparingInt((Interval o) -> {
            return o.start;
        }));
        // Write your code here
        for (int i = 1; i < array.length; i++) {
            Interval pre = array[i - 1];
            Interval now = array[i];
            if (now.start < pre.end) {
                return false;
            }
        }
        return true;
    }


    /**
     * 253 Meeting Rooms II
     *
     * @param intervals: an array of meeting time intervals
     * @return: the minimum number of conference rooms required
     */
    public int minMeetingRooms(List<Interval> intervals) {
        // Write your code here

        if (intervals == null || intervals.size() == 0) {
            return 0;
        }

        PriorityQueue<Integer> pq = new PriorityQueue<>();

        Collections.sort(intervals, new Comparator<Interval>() {

            @Override
            public int compare(Interval a, Interval b) {
                return a.start - b.start;
            }
        });

        pq.offer(intervals.get(0).end);

        for (int i = 1; i < intervals.size(); i++) {
            if (pq.peek() < intervals.get(i).start) {
                pq.poll();
            }
            pq.offer(intervals.get(i).end);
        }
        return pq.size();
    }


}
