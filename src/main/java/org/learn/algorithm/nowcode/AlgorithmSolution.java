package org.learn.algorithm.nowcode;

/**
 * @author luk
 */
public class AlgorithmSolution {


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
        // write code here
        if (s == null || t == null) {
            return "";
        }
        int m = s.length() - 1;
        int n = t.length() - 1;
        int carry = 0;
        StringBuilder builder = new StringBuilder();
        while (m >= 0 || n >= 0 || carry > 0) {
            int val = (m >= 0 ? Character.getNumericValue(s.charAt(m--)) : 0) + (n >= 0 ? Character.getNumericValue(t.charAt(n--)) : 0) + carry;

            builder.append(val % 10);

            carry = val / 10;
        }
        return builder.reverse().toString();
    }


}
