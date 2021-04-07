package org.learn.algorithm.leetcode;

/**
 * 数学理论
 *
 * @author luk
 * @date 2021/4/7
 */
public class MathSolution {


    /**
     * 29. Divide Two Integers
     *
     * @param dividend
     * @param divisor
     * @return
     */
    public int divide(int dividend, int divisor) {
        if (dividend == Integer.MIN_VALUE && divisor == -1) {
            return Integer.MAX_VALUE;
        }
        int sign = ((dividend < 0 && divisor < 0) || (dividend > 0 && divisor > 0)) ? 1 : -1;
        long dvd = Math.abs((long) dividend);
        long dvs = Math.abs((long) divisor);

        int result = 0;

        while (dvd >= dvs) {
            long multi = 1;
            long tmp = dvs;
            while (dvd >= tmp << 1) {
                tmp <<= 1;
                multi <<= 1;
            }
            dvd -= tmp;
            result += multi;
        }
        return result * sign;
    }
}
