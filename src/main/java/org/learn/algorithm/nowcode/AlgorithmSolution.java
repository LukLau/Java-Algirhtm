package org.learn.algorithm.nowcode;

import org.apache.commons.io.FileUtils;
import org.learn.algorithm.datastructure.TreeNode;
import org.springframework.util.ResourceUtils;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

/**
 * @author luk
 */
public class AlgorithmSolution {

    public static void main(String[] args) throws IOException {

        File file = ResourceUtils.getFile("classpath:data.txt");
        List<String> contents = FileUtils.readLines(file, "utf-8");
    }

    /**
     * NC1 大数加法
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * 计算两个数之和
     *
     * @param s string字符串 表示第一个整数
     * @param t string字符串 表示第二个整数
     * @return string字符串
     */
    public String bigAdd(String s, String t) {
        if (s == null || t == null) {
            return "";
        }
        int m = s.length() - 1;
        int n = t.length() - 1;
        StringBuilder builder = new StringBuilder();
        int carry = 0;
        while (m >= 0 || n >= 0 || carry > 0) {
            int val = (m >= 0 ? Character.getNumericValue(s.charAt(m--)) : 0) + (n >= 0 ? Character.getNumericValue(t.charAt(n--)) : 0) + carry;
            carry = val / 10;
            builder.append(val % 10);
        }
        if (builder.length() == 0) {
            return "0";
        }
        return builder.reverse().toString();
    }

    /**
     * NC14 按之字形顺序打印二叉树
     *
     * @param pRoot
     * @return
     */
    public ArrayList<ArrayList<Integer>> Print(TreeNode pRoot) {
        if (pRoot == null) {
            return new ArrayList<>();
        }
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        LinkedList<TreeNode> linkedList = new LinkedList<>();
        linkedList.offer(pRoot);
        boolean leftToRight = true;
        while (!linkedList.isEmpty()) {
            int size = linkedList.size();
            LinkedList<Integer> tmp = new LinkedList<>();
            for (int i = 0; i < size; i++) {
                TreeNode poll = linkedList.poll();
//                int index = leftToRight ? i : size - 1 - i;

                if (leftToRight) {
                    tmp.addLast(poll.val);
                } else {
                    tmp.addFirst(poll.val);
                }
                if (poll.left != null) {
                    linkedList.offer(poll.left);
                }
                if (poll.right != null) {
                    linkedList.offer(poll.right);
                }
            }
            result.add(new ArrayList<>(tmp));
            leftToRight = !leftToRight;
        }
        return result;
    }

    /**
     * 反转字符串
     *
     * @param str string字符串
     * @return string字符串
     */
    public String reverse(String str) {
        // write code here
        if (str == null || str.isEmpty()) {
            return "";
        }
        int len = str.length();
        StringBuilder result = new StringBuilder();
        for (int i = len - 1; i >= 0; i--) {
            result.append(str.charAt(i));
        }
        return result.toString();
    }


    public int Fibonacci(int n) {
        if (n == 0) {
            return 0;
        }
        if (n == 1) {
            return 1;
        }
        int sum1 = 0;
        int sum2 = 1;
        int sum = 0;
        for (int i = 2; i <= n; i++) {
            sum = sum1 + sum2;
            sum1 = sum2;
            sum2 = sum;
        }
        return sum;
    }

}

class Node implements Serializable {
    private String text_payload;

    public String getText_payload() {
        return text_payload;
    }
}
