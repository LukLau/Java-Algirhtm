package org.dora.algorithm.datastructe;

/**
 * @author lauluk
 * @date 2019/03/06
 */
public class TreeNode {
    public int val;
    public TreeNode left;
    public TreeNode right;

    public int previousValue;
    private int currentValue;

    public TreeNode(int x) {
        val = x;
    }
}
