package org.dora.algorithm.swordoffer;

import org.dora.algorithm.datastructe.ListNode;
import org.dora.algorithm.datastructe.TreeNode;

import java.util.ArrayList;
import java.util.LinkedList;

/**
 * @author liulu
 * @date 2019/04/24
 */
public class SwordToOffer {
    /**
     * 二维数组的查找
     *
     * @param target
     * @param array
     * @return
     */
    public boolean Find(int target, int[][] array) {
        if (array == null || array.length == 0) {
            return false;
        }
        int row = array.length;
        int column = array[0].length;
        int i = row - 1;
        int j = 0;
        while (i >= 0 && j < column) {
            if (array[i][j] == target) {
                return true;
            } else if (array[i][j] < target) {
                j++;
            } else {
                i--;
            }
        }
        return false;
    }

    /**
     * 替换空格
     *
     * @param str
     * @return
     */
    public String replaceSpace(StringBuffer str) {
        if (str == null || str.length() == 0) {
            return "";
        }
        String value = str.toString();
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < value.length(); i++) {
            if (value.charAt(i) == ' ') {
                sb.append("%20");
            } else {
                sb.append(value.charAt(i));
            }
        }
        return sb.toString();
    }

    /**
     * 从头到尾打印链表
     *
     * @param listNode
     * @return
     */
    public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
        if (listNode == null) {
            return new ArrayList<>();
        }
        LinkedList<Integer> ans = new LinkedList<>();
        while (listNode != null) {
            ans.addFirst(listNode.val);
            listNode = listNode.next;

        }

        return new ArrayList<>(ans);
    }

    /**
     * 先需
     *
     * @param pre
     * @param in
     * @return
     */
    public TreeNode reConstructBinaryTree(int[] pre, int[] in) {
        if (pre == null || in == null) {
            return null;
        }
        return this.buildPreBinaryTree(0, pre, 0, in.length - 1, in);
    }

    private TreeNode buildPreBinaryTree(int preStart, int[] preorder, int inStart, int inEnd, int[] inOrder) {
        if (preStart >= preorder.length || inStart > inEnd) {
            return null;
        }
        TreeNode root = new TreeNode(preorder[preStart]);
        int index = 0;
        for (int i = inStart; i <= inEnd; i++) {
            if (root.val == inOrder[i]) {
                index = i;
                break;
            }
        }
        root.left = this.buildPreBinaryTree(preStart + 1, preorder, inStart, index - 1, inOrder);
        root.right = this.buildPreBinaryTree(preStart + index - inStart + 1, preorder, index + 1, inEnd, inOrder);
        return root;
    }

    /**
     * 旋转数字的最小数字
     *
     * @param array
     * @return
     */
    public int minNumberInRotateArray(int[] array) {
        if (array == null || array.length == 0) {
            return 0;
        }
        int left = 0;
        int right = array.length - 1;

        // 方案一 和右边界进行比较
//        while (left < right) {
//            int mid = left + (right - left) / 2;
//            if (array[mid] <= array[right]) {
//                right = mid;
//            } else {
//                left = mid + 1;
//            }
//        }

        // 方案二 和左边界进行比较
        while (left < right) {
            if (array[left] < array[right]) {
                return array[left];
            }
            int mid = left + (right - left) / 2;
            if (array[left] <= array[mid]) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return array[left];
    }

    /**
     * 斐波那契数列
     *
     * @param n
     * @return
     */
    public int Fibonacci(int n) {
        if (n == 0) {
            return 0;
        } else if (n <= 2) {
            return 1;
        }
        int sum1 = 1;
        int sum2 = 1;
        int sum3 = 0;
        for (int i = 3; i <= n; i++) {
            sum3 = sum1 + sum2;
            sum1 = sum2;
            sum2 = sum3;
        }
        return sum3;
    }

    /**
     * 跳台阶
     *
     * @param target
     * @return
     */
    public int JumpFloor(int target) {
        if (target == 1) {
            return 1;
        } else if (target == 2) {
            return 2;
        }
        int sum1 = 1;
        int sum2 = 2;
        int sum3 = 0;
        for (int i = 3; i <= target; i++) {
            sum3 = sum1 + sum2;
            sum1 = sum2;
            sum2 = sum3;
        }
        return sum3;
    }

    /**
     * 跳台阶进阶
     *
     * @param target
     * @return
     */
    public int JumpFloorII(int target) {
        if (target == 1) {
            return 1;
        }
        if (target == 2) {
            return 2;
        }

    }


}
