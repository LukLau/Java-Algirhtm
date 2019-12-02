package org.dora.algorithm.geeksforgeek;

import org.dora.algorithm.datastructe.TreeNode;

import java.util.ArrayList;
import java.util.List;

/**
 * @author dora
 * @date 2019/11/29
 */
public class VIP {


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

    public String getRange(int start, int end) {
        if (start == end) {
            return String.valueOf(start);
        }
        return start + "->" + end;
    }


}
