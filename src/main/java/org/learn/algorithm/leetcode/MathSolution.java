package org.learn.algorithm.leetcode;


import org.apache.tomcat.Jar;
import org.springframework.web.servlet.mvc.method.annotation.UriComponentsBuilderMethodArgumentResolver;

import java.awt.image.DataBufferDouble;
import java.util.*;

/**
 * 数学理论
 *
 * @author luk
 * @date 2021/4/7
 */
public class MathSolution {

    public static void main(String[] args) {
        MathSolution solution = new MathSolution();
//        int[] nums = new int[]{5, 4, 4, 3, 2, 1};
//        solution.nthUglyNumber(10);
//        solution.calculate("(1+(4+5+2)-3)+(6+8)");
//        solution.calculateII("3+2*2");
//        System.out.println(solution.numberToWords(12345));

        System.out.println(solution.fullJustifyii(new String[]{"This", "is", "an", "example", "of", "text", "justification."}, 16));
    }

    // 素数相关

    /**
     * todo
     * 204. Count Primes
     *
     * @param n
     * @return
     */
    public int countPrimes(int n) {
        int count = 0;
        boolean[] dp = new boolean[n];
        for (int i = 2; i < n; i++) {
            if (!dp[i]) {
                count++;
                for (int j = 2; j * i < n; j++) {
                    dp[j * i] = true;
                }
            }
        }
        return count;
    }


    /**
     * 29. Divide Two Integers
     *
     * @param dividend
     * @param divisor
     * @return
     */
    public int divide(int dividend, int divisor) {
        if (divisor == -1 && dividend == Integer.MIN_VALUE) {
            return Integer.MAX_VALUE;
        }
        int sign = ((dividend > 0 && divisor > 0) || (dividend < 0 && divisor < 0)) ? 1 : -1;

        long dvd = Math.abs((long) dividend);
        long dvs = Math.abs((long) divisor);
        int result = 0;
        while (dvd >= dvs) {
            long tmp = dvs;
            long multi = 1;
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
    public double myPowII(double x, int n) {
        double result = 1.0;
        long base = Math.abs((long) n);
        while (base != 0) {
            if (base % 2 != 0) {
                result *= x;
            }
            x *= x;
            base = base / 2;
        }
        return n < 0 ? 1 / result : result;
    }

    public double myPow(double x, int n) {
        return internalPow(x, n);
    }

    public double internalPow(double x, long n) {
        if (n == 0) {
            return 1;
        }
        if (n < 0) {
            x = 1 / x;
            n = -n;
        }
        return n % 2 == 0 ? internalPow(x * x, n / 2) : x * internalPow(x * x, n / 2);
    }

    /**
     * todo
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
            return false;
        }
        boolean seenE = false;
        boolean seenNumber = false;
        boolean seenDigit = false;
        boolean seenNumberAfterE = false;
        char[] words = s.toCharArray();
        for (int i = 0; i < words.length; i++) {
            char tmp = words[i];
            if (Character.isDigit(tmp)) {
                seenNumber = true;
                seenNumberAfterE = true;
            } else if (tmp == 'e' || tmp == 'E') {
                if (i == 0 || seenE) {
                    return false;
                }
                if (!Character.isDigit(words[i - 1])) {
                    return false;
                }
                seenE = true;
                seenNumberAfterE = false;
            } else if (tmp == '-' || tmp == '+') {
                if (i > 0 && !(words[i - 1] == 'e' || words[i - 1] == 'E')) {
                    return false;
                }
            } else if (tmp == '.') {
                if (seenDigit) {
                    return false;
                }
                seenDigit = true;
            } else {
                return true;
            }
        }
        return seenNumberAfterE & seenNumber;
    }


    public String addBinary(String a, String b) {
        if (a == null || b == null) {
            return "";
        }
        int m = a.length() - 1;

        int n = b.length() - 1;

        int carry = 0;
        StringBuilder builder = new StringBuilder();
        while (m >= 0 || n >= 0 || carry != 0) {
            int val = (m >= 0 ? Character.getNumericValue(a.charAt(m--)) : 0) + (n >= 0 ? Character.getNumericValue(b.charAt(n--)) : 0) + carry;

            builder.append(val % 2);

            carry = val / 2;
        }
        return builder.reverse().toString();
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
        List<String> result = new ArrayList<>();
        int startIndex = 0;
        while (startIndex < words.length) {
            int endIndex = startIndex;
            int line = 0;
            while (endIndex < words.length && line + words[endIndex].length() <= maxWidth) {
                line += words[endIndex].length() + 1;
                endIndex++;
            }
            boolean lastRow = endIndex == words.length;
            int wordCount = endIndex - startIndex;
            StringBuilder builder = new StringBuilder();
            if (wordCount == 1) {
                builder.append(words[startIndex]);
            } else {
                int blankCount = lastRow ? 1 : 1 + (maxWidth - line + 1) / (wordCount - 1);
                int extraCount = lastRow ? 0 : (maxWidth - line + 1) % (wordCount - 1);
                String constructRow = constructRow(words, startIndex, endIndex, blankCount, extraCount);

                builder.append(constructRow);
            }
            result.add(trimRow(builder.toString(), maxWidth));

            startIndex = endIndex;
        }
        return result;
    }

    private String trimRow(String tmp, int maxWidth) {
        while (tmp.length() > maxWidth) {
            tmp = tmp.substring(0, tmp.length() - 1);
        }
        while (tmp.length() < maxWidth) {
            tmp = tmp + " ";
        }
        return tmp;
    }

    private String constructRow(String[] words, int startIndex, int endIndex, int blankCount, int extraCount) {
        StringBuilder builder = new StringBuilder();
        for (int i = startIndex; i < endIndex; i++) {
            int tmp = blankCount;
            builder.append(words[i]);
            while (tmp-- > 0) {
                builder.append(" ");
            }
            if (extraCount-- > 0) {
                builder.append(" ");
            }
        }
        return builder.toString();
    }

    public List<String> fullJustifyii(String[] words, int maxWidth) {
        if (words == null || words.length == 0) {
            return new ArrayList<>();
        }
        int startIndex = 0;
        List<String> result = new ArrayList<>();
        while (startIndex < words.length) {
            int line = 0;
            int index = 0;
            while (startIndex + index < words.length && line + words[startIndex + index].length() <= maxWidth - index) {
                line += words[startIndex + index].length();
                index++;
            }
            StringBuilder builder = new StringBuilder();
            boolean lastRow = startIndex + index == words.length;
            for (int i = 0; i < index; i++) {
                builder.append(words[startIndex + i]);
                if (lastRow) {
                    builder.append(" ");
                } else if (i != index - 1) {
                    int blankCount = (maxWidth - line) / (index - 1) + (i < (maxWidth - line) % (index - 1) ? 1 : 0);
                    while (blankCount-- > 0) {
                        builder.append(" ");
                    }
                }
            }
            result.add(trimRow(builder.toString(), maxWidth));
            startIndex = startIndex + index;
        }
        return result;
    }

    /**
     * 69. Sqrt(x)
     *
     * @param x
     * @return
     */
    public int mySqrt(int x) {
        double precision = 0.00001;
        double result = x;
        while (result * result - x > precision) {
            result = (result + x / result) / 2;
        }
        return (int) result;
    }


    /**
     * 89. Gray Code
     * todo
     *
     * @param n
     * @return
     */
    public List<Integer> grayCode(int n) {
        List<Integer> result = new ArrayList<>();
        int count = (int) Math.pow(2, n);
        for (int i = 0; i < count; i++) {
            int val = (i >> 1) ^ i;
            result.add(val);
        }
        return result;
    }

    public int longestConsecutive(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        Map<Integer, Integer> map = new HashMap<>();
        int result = 0;
        for (int num : nums) {
            if (!map.containsKey(num)) {
                Integer left = map.getOrDefault(num - 1, 0);
                Integer right = map.getOrDefault(num + 1, 0);

                int val = left + right + 1;
                result = Math.max(result, val);

                map.put(num - left, val);
                map.put(num + right, val);
                map.put(num, val);
            }
        }
        return result;
    }


    /**
     * todo
     * 149. Max Points on a Line
     *
     * @param points
     * @return
     */
    public int maxPoints(int[][] points) {
        if (points == null || points.length == 0) {
            return 0;
        }
        int result = 0;
        for (int i = 0; i < points.length; i++) {
            int number = 0;
            int overlap = 0;
            Map<Integer, Map<Integer, Integer>> map = new HashMap<>();
            for (int j = i + 1; j < points.length; j++) {
                int x = points[j][0] - points[i][0];
                int y = points[j][1] - points[i][1];

                if (x == 0 && y == 0) {
                    overlap++;
                    continue;
                }
                int gcd = gcd(x, y);

                x /= gcd;
                y /= gcd;

                Map<Integer, Integer> tmp = map.getOrDefault(x, new HashMap<>());
                Integer count = tmp.getOrDefault(y, 0);

                count++;

                tmp.put(y, count);

                map.put(x, tmp);

                number = Math.max(number, count);
            }
            result = Math.max(result, 1 + number + overlap);
        }
        return result;
    }

    private int gcd(int a, int b) {
        if (b == 0) {
            return a;
        }
        return gcd(b, a % b);
    }


    /**
     * 152. Maximum Product Subarray
     *
     * @param nums
     * @return
     */
    public int maxProduct(int[] nums) {
        int max = nums[0];
        int min = nums[0];
        int result = nums[0];
        for (int i = 1; i < nums.length; i++) {
            int val = nums[i];
            int tmpMax = Math.max(Math.max(val * max, min * val), val);
            int tmpMin = Math.min(Math.min(val * max, min * val), val);
            result = Math.max(result, tmpMax);
            max = tmpMax;
            min = tmpMin;
        }
        return result;
    }

    // 位运算相关

    // 寻找唯一的一个数 //

    /**
     * 136. Single Number
     *
     * @param nums
     * @return
     */
    public int singleNumber(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int result = 0;
        for (int num : nums) {
            result ^= num;
        }
        return result;
    }


    /**
     * todo
     * 137. Single Number II
     *
     * @param nums
     * @return
     */
    public int singleNumberII(int[] nums) {
        int result = 0;
        for (int i = 0; i < 32; i++) {
            int sum = 0;
            for (int j = 0; j < nums.length; j++) {
                sum += (nums[j] >> i) & 1;
            }
            result |= ((sum % 3) << i);
        }
        return result;
    }


    /**
     * 260. Single Number III
     *
     * @param nums
     * @return
     */
    public int[] singleNumberIII(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new int[]{};
        }
        int base = 0;
        for (int num : nums) {
            base ^= num;
        }
        base &= -base;
        int[] result = new int[2];
        for (int num : nums) {
            if ((num & base) != 0) {
                result[0] ^= num;
            } else {
                result[1] ^= num;
            }
        }
        return result;
    }


    public int[] singleNumberIIIV2(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new int[]{};
        }
        int base = 0;
        for (int num : nums) {
            base ^= num;
        }
        int index = 0;
        for (int i = 0; i < 32; i++) {
            if ((base & (1 << i)) != 0) {
                index = i;
                break;
            }
        }
        int[] result = new int[2];
        for (int num : nums) {
            if ((num & (1 << index)) == 0) {
                result[0] ^= num;
            } else {
                result[1] ^= num;
            }
        }
        return result;
    }


    /**
     * 190. Reverse Bits
     *
     * @param n
     * @return
     */
    public int reverseBits(int n) {
        int result = 0;
        for (int i = 0; i < 32; i++) {
            result <<= 1;
            boolean remain = (n & (1 << i)) != 0;

            if (remain) {
                result++;
            }
        }
        return result;
    }


    /**
     * 191. Number of 1 Bits
     *
     * @param n
     * @return
     */
    public int hammingWeight(int n) {
        int count = 0;
        while (n != 0) {
            count++;
            n &= (n - 1);
        }
        return count;
    }


    /**
     * 231. Power of Two
     *
     * @param n
     * @return
     */
    public boolean isPowerOfTwo(int n) {
        return n > 0 && (n & (n - 1)) == 0;
    }


    /**
     * 201. Bitwise AND of Numbers Range
     * todo
     *
     * @param m
     * @param n
     * @return
     */
    public int rangeBitwiseAnd(int m, int n) {
        while (m < n) {
            n = n & (n - 1);
        }
        return n;
    }


    public int rangeBitwiseAndV2(int m, int n) {
        int shiftCount = 0;
        while (m != n) {
            m >>= 1;
            n >>= 1;
            shiftCount++;
        }
        return m << shiftCount;
    }

    /**
     * 202. Happy Number
     *
     * @param n
     * @return
     */
    public boolean isHappy(int n) {
        Set<Integer> used = new HashSet<>();
        while (true) {
            if (used.contains(n)) {
                return false;
            }
            int sum = 0;
            int tmp = n;
            while (tmp != 0) {
                int mod = tmp % 10;
                sum += mod * mod;
                tmp /= 10;
            }
            if (sum == 1) {
                return true;
            }
            n = sum;
            used.add(n);
        }
    }


    /**
     * 233. Number of Digit One
     * todo 需要数学规律
     *
     * @param n
     * @return
     */
    public int countDigitOne(int n) {
        String s = String.valueOf(n);
        return -1;
    }


    // 摩尔投票法

    /**
     * 169. Majority Element
     *
     * @param nums
     * @return
     */
    public int majorityElement(int[] nums) {
        int candidate = nums[0];
        int count = 0;
        for (int num : nums) {
            if (num == candidate) {
                count++;
                continue;
            }
            count--;
            if (count == 0) {
                candidate = num;
                count = 1;
            }
        }
        return candidate;
    }


    /**
     * 229. Majority Element II
     *
     * @param nums
     * @return
     */
    public List<Integer> majorityElementII(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new ArrayList<>();
        }
        List<Integer> result = new ArrayList<>();
        int candidateA = nums[0];
        int candidateB = nums[0];
        int countA = 0;
        int countB = 0;
        for (int tmp : nums) {
            if (tmp == candidateA) {
                countA++;
                continue;
            }
            if (tmp == candidateB) {
                countB++;
                continue;
            }
            if (countA == 0) {
                candidateA = tmp;
                countA = 1;
                continue;
            }
            if (countB == 0) {
                candidateB = tmp;
                countB = 1;
                continue;
            }
            countA--;
            countB--;

        }
        countA = 0;
        countB = 0;
        for (int tmp : nums) {
            if (tmp == candidateA) {
                countA++;
            } else if (tmp == candidateB) {
                countB++;
            }
        }

        if (3 * countA >= nums.length) {
            result.add(candidateA);
        }
        if (3 * countB >= nums.length) {
            result.add(candidateB);
        }
        return result;
    }


    // 逆波兰数, 计算器 //

    /**
     * 150. Evaluate Reverse Polish Notation
     *
     * @param tokens
     * @return
     */
    public int evalRPN(String[] tokens) {
        if (tokens == null || tokens.length == 0) {
            return 0;
        }
        Stack<Integer> stack = new Stack<>();
        for (String token : tokens) {
            if ("+".equals(token)) {
                stack.push(stack.pop() + stack.pop());
            } else if ("-".equals(token)) {
                stack.push(-1 * stack.pop() + stack.pop());
            } else if ("*".equals(token)) {
                stack.push(stack.pop() * stack.pop());
            } else if ("/".equals(token)) {
                int second = stack.pop();
                int first = stack.pop();
                stack.push(first / second);
            } else {
                stack.push(Integer.parseInt(token));
            }
        }
        return stack.pop();
    }

    /**
     * 224. Basic Calculator
     *
     * @param s
     * @return
     */
    public int calculate(String s) {
        if (s == null) {
            return 0;
        }
        s = s.trim();
        if (s.isEmpty()) {
            return 0;
        }
        char[] words = s.toCharArray();
        Stack<Integer> stack = new Stack<>();
        char sign = '+';
        int index = 0;
        while (index < words.length) {
            char tmp = words[index];
            if (tmp == '(') {
                int endIndex = index;

                int count = 0;
                while (endIndex < words.length) {
                    if (words[endIndex] != '(' && words[endIndex] != ')') {
                        endIndex++;
                        continue;
                    }
                    if (words[endIndex] == '(') {
                        count++;
                    }
                    if (words[endIndex] == ')') {
                        count--;
                    }
                    if (count == 0) {
                        break;
                    }
                    endIndex++;
                }
                stack.push(calculate(s.substring(index + 1, endIndex)));
                index = endIndex + 1;
            } else if (Character.isDigit(tmp)) {
                int value = 0;
                while (index < words.length && Character.isDigit(words[index])) {
                    value = value * 10 + Character.getNumericValue(words[index]);
                    index++;
                }
                stack.push(value);
            }
            if (index == words.length || words[index] != ' ') {
                if (sign == '-') {
                    stack.push(-1 * stack.pop());
                } else if (sign == '*') {
                    stack.push(stack.pop() * stack.pop());
                } else if (sign == '/') {
                    Integer dividend = stack.pop();
                    Integer divisor = stack.pop();
                    stack.push(divisor / dividend);
                }
                if (index != words.length) {
                    sign = words[index];
                }
            }
            index++;
        }
        int result = 0;
        for (int tmp : stack) {
            result += tmp;
        }
        return result;
    }


    /**
     * 227. Basic Calculator II
     *
     * @param s
     * @return
     */
    public int calculateII(String s) {
        if (s == null) {
            return 0;
        }
        s = s.trim();
        if (s.isEmpty()) {
            return 0;
        }
        char[] words = s.toCharArray();

        int index = 0;
        char sign = '+';
        int result = 0;
        Stack<Integer> stack = new Stack<>();
        while (index < words.length) {

            if (Character.isDigit(words[index])) {
                int tmpValue = 0;
                while (index < words.length && Character.isDigit(words[index])) {
                    tmpValue = tmpValue * 10 + Character.getNumericValue(words[index++]);
                }
                stack.push(tmpValue);
            }
            if (index == words.length || words[index] != ' ') {
                if (sign == '-') {
                    stack.push(-1 * stack.pop());
                } else if (sign == '*') {
                    stack.push(stack.pop() * stack.pop());
                } else if (sign == '/') {
                    int dividend = stack.pop();
                    int divisor = stack.pop();
                    stack.push(divisor / dividend);
                }
                if (index != words.length) {
                    sign = words[index];
                }
            }
            index++;
        }
        for (Integer tmp : stack) {
            result += tmp;
        }
        return result;
    }


    /**
     * @param s: the expression string
     * @return: the answer
     */
    public int calculateIII(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        }
        int startIndex = 0;
        int len = s.length();
        char sign = '+';
        Stack<Integer> stack = new Stack<>();
        while (startIndex <= len) {
            if (startIndex < len && s.charAt(startIndex) == '(') {
                int tmpIndex = startIndex;
                int count = 0;
                while (tmpIndex < len) {
                    char current = s.charAt(tmpIndex);
                    if (current != '(' && current != ')') {
                        tmpIndex++;
                        continue;
                    }
                    if (current == '(') {
                        count++;
                    }
                    if (current == ')') {
                        count--;
                    }
                    if (count == 0) {
                        break;
                    }
                    tmpIndex++;
                }
                int val = calculateIII(s.substring(startIndex + 1, tmpIndex));
                stack.push(val);
                startIndex = tmpIndex;
            }
            if (startIndex < len && Character.isDigit(s.charAt(startIndex))) {
                int tmp = 0;
                while (startIndex < len && Character.isDigit(s.charAt(startIndex))) {
                    tmp = tmp * 10 + Character.getNumericValue(s.charAt(startIndex++));
                }
                stack.push(tmp);
            }
            if (startIndex == len || s.charAt(startIndex) != ' ') {
                if (sign == '-') {
                    stack.push(-1 * stack.pop());
                } else if (sign == '*') {
                    stack.push(stack.pop() * stack.pop());
                } else if (sign == '/') {
                    int second = stack.pop();
                    int first = stack.pop();
                    stack.push(first / second);
                }
                if (startIndex != len) {
                    sign = s.charAt(startIndex);
                }
            }
            startIndex++;
        }
        int result = 0;
        for (Integer tmp : stack) {
            result += tmp;
        }
        return result;
    }


    // 丑数相关


    /**
     * 263. Ugly Number
     *
     * @param n
     * @return
     */
    public boolean isUgly(int n) {
        if (n <= 0) {
            return false;
        }
        if (n < 7) {
            return true;
        }
        while (true) {
            if (n == 2 || n == 3 || n == 5) {
                return true;
            }
            if (n % 2 == 0) {
                n /= 2;
            } else if (n % 3 == 0) {
                n /= 3;
            } else if (n % 5 == 0) {
                n /= 5;
            } else {
                return false;
            }
        }
    }


    public boolean isUglyV2(int num) {
        while (num > 0 && num % 2 == 0) {
            num /= 2;
        }
        while (num > 0 && num % 3 == 0) {
            num /= 3;
        }
        while (num > 0 && num % 5 == 0) {
            num /= 5;
        }
        return num == 1;
    }


    /**
     * 264. Ugly Number II
     *
     * @param n
     * @return
     */
    public int nthUglyNumber(int n) {
        if (n < 7) {
            return n;
        }
        int index2 = 0;
        int index3 = 0;
        int index5 = 0;
        int[] result = new int[n];
        result[0] = 1;
        int index = 1;
        while (index < n) {
            int val = Math.min(Math.min(result[index2] * 2, result[index3] * 3), result[index5] * 5);
            if (val == result[index2] * 2) {
                index2++;
            }
            if (val == result[index3] * 3) {
                index3++;
            }
            if (val == result[index5] * 5) {
                index5++;
            }
            result[index] = val;

            index++;
        }
        return result[n - 1];
    }


    /**
     * todo
     * 274. H-Index
     *
     * @param citations
     * @return
     */
    public int hIndex(int[] citations) {
        if (citations == null || citations.length == 0) {
            return 0;
        }
        return 0;

    }


    /**
     * todo
     * 319
     * Bulb Switcher
     *
     * @param n: a Integer
     * @return: how many bulbs are on after n rounds
     */
    public int bulbSwitch(int n) {
        // Write your code here
        return -1;
    }


    /**
     * WC61 删除元素
     *
     * @param A
     * @param elem
     * @return
     */
    public int removeElement(int[] A, int elem) {
        if (A == null || A.length == 0) {
            return 0;
        }
        int index = 0;
        for (int i = 0; i < A.length; i++) {
            if (A[i] != elem) {
                A[index++] = A[i];
            }
        }
        return index;
    }


    /**
     * WC62 有序数组删除重复数字
     *
     * @param A
     * @return
     */
    public int removeDuplicates(int[] A) {
        if (A == null || A.length == 0) {
            return 0;
        }
        int index = 1;
        for (int i = 1; i < A.length; i++) {
            if (A[i] != A[i - 1]) {
                A[index++] = A[i];
            }
        }
        return index;
    }

    // 阿拉伯数字转中文


    public String convertToArab(int nums) {
        String[] one = new String[]{"零", "一", "二", "三", "四", "五", "六", "七", "八", "九"};
        String[] two = new String[]{"", "十", "百", "千"};
        String[] three = new String[]{"", "万", "亿", "万亿"};
        String result = "";
        while (nums != 0) {
            int remain = nums % 10000;

            String desc = getDesc(remain, one, two);

            if (nums / 10000 != 0) {
                result = three[remain / 10000] + result;
            }
            result += desc;

            nums /= 10000;
        }
        return result;
    }

    private String getDesc(int remain, String[] one, String[] two) {
        if (remain == 0) {
            return "";
        }
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < 4; i++) {

        }
        return null;
    }

    /**
     * 172. Factorial Trailing Zeroes
     *
     * @param n
     * @return
     */
    public int trailingZeroes(int n) {
        int count = 0;
        while (n / 5 != 0) {
            count += n / 5;
        }
        return count;

    }


    /**
     * 258. Add Digits
     * todo
     *
     * @param num
     * @return
     */
    public int addDigits(int num) {
        return -1;

    }


    // --数字转化为阿拉伯， 汉子 -- //

    /**
     * 273. Integer to English Words
     * 太艰难了  终于pass了
     *
     * @param num
     * @return
     */
    public String numberToWords(int num) {
        if (num == 0) {
            return "Zero";
        }
        String[] belowTwenty = new String[]{"Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve",
                "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"};
        String[] moreThanTwenty = new String[]{"", "Ten", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"};
        String[] moreThousand = new String[]{"", "Thousand", "Million", "Billion"};
        List<String> result = new ArrayList<>();

        int index = 0;

        while (num > 0) {

            int remain = num % 1000;

            String read = generateWord(remain, belowTwenty, moreThanTwenty);

            if (index > 0 && !read.isEmpty()) {
                read = read + " " + moreThousand[index];
            }
            if (!read.isEmpty()) {
                result.add(read);
            }

            index++;

            num /= 1000;
        }
        StringBuilder builder = new StringBuilder();

        int len = result.size();
        for (int i = len - 1; i >= 0; i--) {
            builder.append(result.get(i));
            if (i != 0) {
                builder.append(" ");
            }
        }
        return builder.toString();
    }

    private String generateWord(int num, String[] belowTwenty, String[] moreThanTwenty) {
        StringBuilder builder = new StringBuilder();

        String threeNum = belowTwenty[num / 100];

        if (!"Zero".equals(threeNum)) {
            builder.append(threeNum).append(" Hundred");
        }
        int twoNum = num % 100;

        if (twoNum < 20 && twoNum > 0) {
            builder.append(" ").append(belowTwenty[twoNum % 20]);
        } else {
            builder.append(" ").append(moreThanTwenty[twoNum / 10]);
            int oneNum = twoNum % 10;
            if (oneNum > 0) {
                builder.append(" ").append(belowTwenty[oneNum]);
            }
        }
        return builder.toString().trim();
    }

    // 重复数字相关问题 //


    /**
     * https://leetcode.com/problems/find-the-duplicate-number/solution/
     * <p>
     * 287. Find the Duplicate Number
     *
     * @param nums
     * @return
     */
    public int findDuplicate(int[] nums) {
        Arrays.sort(nums);

        for (int i = 1; i < nums.length; i++) {
            if (nums[i] == nums[i - 1]) {
                return nums[i];
            }
        }
        return -1;
    }


}
