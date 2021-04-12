package org.learn.algorithm.leetcode;

import java.util.ArrayList;
import java.util.List;

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


    /**
     * 50. Pow(x, n)
     *
     * @param x
     * @param n
     * @return
     */
    public double myPow(double x, int n) {
        double result = 1.0;
        long p = Math.abs((long) n);
        while (p != 0) {
            if (p % 2 != 0) {
                result *= x;
            }
            x *= x;
            p >>= 1;
        }
        return n < 0 ? 1 / result : result;
    }

    /**
     * 65. Valid Number
     *
     * @param s
     * @return
     */
    public boolean isNumber(String s) {
        if (s == null) {
            return false;
        }
        s = s.trim();
        if (s.isEmpty()) {
            return true;
        }
        boolean seenNumber = false;
        boolean seenNumberAfterE = true;
        boolean seenDigit = false;
        boolean seenE = false;
        char[] words = s.toCharArray();
        for (int i = 0; i < words.length; i++) {
            char tmp = words[i];
            if (Character.isDigit(tmp)) {
                seenNumber = true;
                seenNumberAfterE = true;
            } else if (tmp == 'e' || tmp == 'E') {
                if (seenE || i == 0) {
                    return false;
                }
                if (!seenNumber) {
                    return false;
                }
                seenE = true;
                seenNumberAfterE = false;
            } else if (tmp == '-' || tmp == '+') {
                if (i > 0 && (words[i - 1] != 'e' && words[i - 1] != 'E')) {
                    return false;
                }
            } else if (tmp == '.') {
                if (seenE || seenDigit) {
                    return false;
                }
                seenDigit = true;
            } else {
                return false;
            }
        }
        return seenNumber && seenNumberAfterE;
    }


    /**
     * todo
     * 68. Text Justification
     *
     * @param words
     * @param maxWidth
     * @return
     */
    public List<String> fullJustify(String[] words, int maxWidth) {
        if (words == null || words.length == 0) {
            return new ArrayList<>();
        }
        int startIndex = 0;
        while (startIndex < words.length) {
            int endIndex = startIndex;
            int line = 0;
            while (endIndex < words.length && line + words[endIndex].length() <= maxWidth) {
                line += words[endIndex].length() + 1;
                endIndex++;
            }
            boolean lastRow = endIndex == words.length;
            int blankSpace = maxWidth - line + 1;
            int wordCount = endIndex - startIndex;
            StringBuilder builder = new StringBuilder();
            if (wordCount == 1) {
                builder.append(words[startIndex]);
            } else {


            }


            startIndex = endIndex;
        }
        return new ArrayList<>();
    }

    /**
     * 69. Sqrt(x)
     *
     * @param x
     * @return
     */
    public int mySqrt(int x) {
        double precision = 0.00001;
        int result = x;
        while (result * result - x > precision) {
            result = (result + x / result) / 2;
        }
        return result;
    }


    /**
     * 89. Gray Code
     * todo
     *
     * @param n
     * @return
     */
    public List<Integer> grayCode(int n) {
        return null;
    }


    /**
     * todo
     * 91. Decode Ways
     *
     * @param s
     * @return
     */
    public int numDecodings(String s) {
        if (s.startsWith("0")) {
            s = s.substring(0, s.length() - 1);
        }
        int m = s.length();
        StringBuilder builder = new StringBuilder();
        for (int i = m - 1; i >= 0; i--) {
            char tmp = s.charAt(i);
            char t = (char) ((tmp - '1') % 26 + 'A');
            builder.append(t);
        }
        return builder.length();
    }


}
