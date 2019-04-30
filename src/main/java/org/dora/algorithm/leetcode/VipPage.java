package org.dora.algorithm.leetcode;

import org.dora.algorithm.datastructe.TreeNode;

/**
 * @author dora
 * @date 2019-04-30
 */
public class VipPage {


    /**
     * 156、Binary Tree Upside Down
     * <a href="https://www.lintcode.com/problem/binary-tree-upside-down/description">
     * 从低到上旋转树</a>
     *
     * @param root
     * @return
     */
    public TreeNode upsideDownBinaryTree(TreeNode root) {
        // write your code here
        if (root == null || root.left == null) {
            return root;
        }

        TreeNode left = root.left;

        TreeNode right = root.right;

        TreeNode ans = this.upsideDownBinaryTree(left);

        left.left = right;

        left.right = root;

        root.left = null;

        root.right = null;
        return ans;

    }


    /**
     * 159、Longest Substring with At Most Two Distinct Characters
     * todo 滑动窗口 不懂
     * @param s
     * @return
     */
    public int lengthOfLongestSubstringTwoDistinct(String s) {
        int left = 0, right = -1, res = 0;
        for (int i = 1; i < s.length(); ++i) {
            if (s[i] == s[i - 1]) {
                continue;
            }
            if (right >= 0 && s[right] != s[i]) {
                res = Math.max(res, i - left);
                left = right + 1;
            }
            right = i - 1;
        }
        return Math.max(s.length() - left, res);
    }


}
