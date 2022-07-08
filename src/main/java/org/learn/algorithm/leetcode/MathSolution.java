package org.learn.algorithm.leetcode;


import java.nio.file.StandardWatchEventKinds;
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
        solution.grayCode(2);
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
        if (divisor == 0) {
            return Integer.MAX_VALUE;
        }
        long result = 0;
        long dvd = Math.abs((long) dividend);
        long dvs = Math.abs((long) divisor);
        int sign = ((dividend > 0 && divisor > 0) || (dividend < 0 && divisor < 0)) ? 1 : -1;
        while (dvd >= dvs) {
            long tmp = dvs;
            long multi = 1;
            while (dvd >= (tmp << 1)) {
                tmp <<= 1;
                multi <<= 1;
            }
            dvd -= tmp;
            result += multi;
        }
        return (int) (result * sign);
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
        long num = Math.abs((long) n);
        while (num != 0) {
            if (num % 2 != 0) {
                result *= x;
            }
            x *= x;
            num >>= 1;
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
            n = -n;
            x = 1 / x;
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
        if (s == null || s.isEmpty()) {
            return true;
        }
        boolean seenE = true;
        boolean seenSign = false;
        boolean seenNumber = true;
        boolean seenEAfterNumber = true;
        boolean seenDigit = false;
        char[] words = s.toCharArray();
        int endIndex = 0;
        while (endIndex < words.length) {
            char tmp = words[endIndex];
            if (Character.isDigit(tmp)) {
                seenNumber = true;
                seenEAfterNumber = true;
            }
        }
        return false;
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
            int line = 0;
            int endIndex = startIndex;
            while (endIndex < words.length && line + words[endIndex].length() <= maxWidth) {
                line += words[endIndex].length() + 1;
                endIndex++;
            }
            boolean lastRow = endIndex == words.length;
            int workCount = endIndex - startIndex;
            StringBuilder tmp = new StringBuilder();
            if (workCount == 1) {
                tmp.append(words[startIndex]);
            } else {
                int blankCount = lastRow ? 1 : 1 + (maxWidth - line + 1) / (workCount - 1);
                int extraBlank = lastRow ? 0 : (maxWidth - line + 1) % (workCount - 1);
                tmp.append(constructRow(words, startIndex, endIndex, blankCount, extraBlank));
            }
            result.add(trimRow(tmp.toString(), maxWidth));
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
        int num = 1 << n;
        for (int i = 0; i < num; i++) {
            int tmp = (i >> 1) ^ i;
            result.add(tmp);
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
            Map<Integer, Map<Integer, Integer>> map = new HashMap<>();
            int overlap = 0;
            int number = 0;
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
                tmp.put(y, count + 1);
                map.put(x, tmp);
                number = Math.max(number, map.get(x).get(y));
            }
            result = Math.max(result, 1 + overlap + number);
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
            if ((n & 1) != 0) {
                result++;
            }
            n >>= 1;
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


    /**
     * @param nums: an array containing n + 1 integers which is between 1 and n
     * @return: the duplicate one
     */
    public int findDuplicate(int[] nums) {
        // write your code here
        int base = 0;
        for (int num : nums) {
            int remain = num & Integer.MAX_VALUE;

            boolean exist = (base & (1 << remain)) != 0;
            if (exist) {
                return num;
            }
            base = base ^ (1 << remain);
        }
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
            } else {
                count--;
                if (count == 0) {
                    candidate = num;
                    count = 1;
                }
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
        int candidateA = nums[0];
        int candidateB = nums[0];

        int countA = 0;
        int countB = 0;

        for (int number : nums) {
            if (number == candidateA) {
                countA++;
                continue;
            }
            if (number == candidateB) {
                countB++;
                continue;
            }
            if (countA == 0) {
                candidateA = number;
                countA = 1;
                continue;
            }
            if (countB == 0) {
                candidateB = number;
                countB = 1;
                continue;
            }
            countA--;
            countB--;
        }
        countA = 0;
        countB = 0;
        for (int num : nums) {
            if (num == candidateA) {
                countA++;
            } else if (num == candidateB) {
                countB++;
            }
        }
        List<Integer> result = new ArrayList<>();
        if (3 * countA > nums.length) {
            result.add(candidateA);
        }
        if (3 * countB > nums.length) {
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
        if (s == null || s.isEmpty()) {
            return 0;
        }
        int len = s.length();
        int result = 0;
        int sign = 1;
        int endIndex = 0;
        Stack<Integer> stack = new Stack<>();
        while (endIndex < len) {
            if (Character.isDigit(s.charAt(endIndex))) {
                int tmp = 0;
                while (endIndex < len && Character.isDigit(s.charAt(endIndex))) {
                    tmp = tmp * 10 + Character.getNumericValue(s.charAt(endIndex++));
                }
                result += sign * tmp;
            }
            if (endIndex != len && s.charAt(endIndex) != ' ') {
                if (s.charAt(endIndex) != ' ') {
                    if (s.charAt(endIndex) == '(') {
                        stack.push(result);
                        stack.push(sign);
                        result = 0;
                        sign = 1;
                    } else if (s.charAt(endIndex) == ')') {
                        result = result * stack.pop() + stack.pop();
                    } else if (s.charAt(endIndex) == '+') {
                        sign = 1;
                    } else {
                        sign = -1;
                    }
                }
            }
            endIndex++;
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
        if (s == null || s.isEmpty()) {
            return 0;
        }
        Stack<Integer> stack = new Stack<>();
        int tmp = 0;
        int len = s.length();
        int startIndex = 0;
        char sign = '+';
        while (startIndex <= len) {
            if (startIndex < len && Character.isDigit(s.charAt(startIndex))) {
                while (startIndex < len && Character.isDigit(s.charAt(startIndex))) {
                    tmp = tmp * 10 + Character.getNumericValue(s.charAt(startIndex++));
                }
            }
            if (startIndex == len || s.charAt(startIndex) != ' ') {
                if (sign == '+') {
                    stack.push(tmp);
                }
                if (sign == '-') {
                    stack.push(-tmp);
                } else if (sign == '*') {
                    stack.push(stack.pop() * tmp);
                } else if (sign == '/') {
                    stack.push(stack.pop() / tmp);
                }
                tmp = 0;
                if (startIndex != len) {
                    sign = s.charAt(startIndex);
                }
            }
            startIndex++;
        }
        int result = 0;
        for (Integer num : stack) {
            result += num;
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
}
