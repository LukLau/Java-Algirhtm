package org.dora.algorithm.geeksforgeek;

/**
 * @author dora
 * @date 2019/11/4
 */
public class MathematicalAlgorithm {

    /**
     * 牛顿平方法来解决
     * 69. Sqrt(x)
     *
     * @param x
     * @return
     */
    public int mySqrt(int x) {
        double precision = 0.00001;

        double result = x;

        while ((result * result - x) > precision) {
            result = (result + x / result) / 2;
        }
        return (int) result;
    }
}
