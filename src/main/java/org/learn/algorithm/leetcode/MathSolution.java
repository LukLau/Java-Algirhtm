package org.learn.algorithm.leetcode;


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
        int[] nums = new int[]{5, 4, 4, 3, 2, 1};
        solution.findDuplicate(nums);
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
                    dp[i * j] = true;
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
        if (dividend == Integer.MIN_VALUE && divisor == -1) {
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
            int blankSpace = maxWidth - line + 1;
            int wordCount = endIndex - startIndex;
            StringBuilder builder = new StringBuilder();
            if (wordCount == 1) {
                builder.append(words[startIndex]);
            } else {
                int space = lastRow ? 1 : 1 + blankSpace / (wordCount - 1);
                int extra = lastRow ? 0 : blankSpace % (wordCount - 1);
                builder.append(constructRow(words, startIndex, endIndex, space, extra));
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
            int overflow = 0;
            Map<Integer, Map<Integer, Integer>> map = new HashMap<>();
            int count = 0;
            for (int j = i + 1; j < points.length; j++) {
                int x = points[j][0] - points[i][0];
                int y = points[j][1] - points[i][1];

                if (x == 0 && y == 0) {
                    overflow++;
                    continue;
                }
                int gcd = gcd(x, y);

                x /= gcd;

                y /= gcd;

                if (!map.containsKey(x)) {
                    Map<Integer, Integer> tmp = new HashMap<>();
                    tmp.put(y, 1);
                    map.put(x, tmp);
                } else {
                    Map<Integer, Integer> tmp = map.get(x);

                    Integer num = tmp.getOrDefault(y, 0);

                    tmp.put(y, num + 1);
                }
                count = Math.max(count, map.get(x).get(y));
            }
            result = Math.max(result, count + 1 + overflow);
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
            int tmpMax = Math.max(Math.max(max * nums[i], min * nums[i]), nums[i]);
            int tmpMin = Math.min(Math.min(max * nums[i], min * nums[i]), nums[i]);
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
        int[] result = new int[2];
        int base = 0;
        for (int num : nums) {
            base ^= num;
        }
        int index = 0;
        for (int i = 0; i < 32; i++) {
            if ((base & 1 << i) != 0) {
                index = i;
                break;
            }
        }
        for (int num : nums) {
            if ((num & (1 << index)) != 0) {
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
            if ((n & 1) == 1) {
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
        return -1;
    }


    /**
     * 233. Number of Digit One
     * todo
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
        List<Integer> result = new ArrayList<>();
        int candidateA = nums[0];
        int candidateB = nums[0];
        int countA = 0;
        int countB = 0;
        for (int num : nums) {
            if (num == candidateA) {
                countA++;
                continue;
            }
            if (num == candidateB) {
                countB++;
                continue;
            }
            if (countA == 0) {
                countA = 1;
                candidateA = num;
                continue;
            }
            if (countB == 0) {
                countB = 1;
                candidateB = num;
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
                Integer first = stack.pop();
                Integer second = stack.pop();
                stack.push(second - first);
            } else if ("*".equals(token)) {
                stack.push(stack.pop() * stack.pop());
            } else if ("/".equals(token)) {
                Integer first = stack.pop();
                Integer second = stack.pop();
                stack.push(second / first);
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
        int sign = 1;
        char[] words = s.toCharArray();
        int endIndex = 0;
        int result = 0;
        Stack<Integer> stack = new Stack<>();
        while (endIndex < words.length) {
            if (Character.isDigit(words[endIndex])) {
                int tmp = 0;
                while (endIndex < words.length && Character.isDigit(words[endIndex])) {
                    tmp = tmp * 10 + Character.getNumericValue(words[endIndex]);
                    endIndex++;
                }
                result += tmp * sign;
            } else {
                if (words[endIndex] == '+' || words[endIndex] == '-') {
                    sign = words[endIndex] == '+' ? 1 : -1;
                }
                if (words[endIndex] == '(') {
                    stack.push(result);
                    stack.push(sign);
                    sign = 1;
                    result = 0;
                }
                if (words[endIndex] == ')') {
                    result = result * stack.pop() + stack.pop();
                }
                endIndex++;
            }
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

        // trim blank space
        if (s.isEmpty()) {
            return 0;
        }
        Stack<Integer> stack = new Stack<>();
        int sign;

        // virtual prefix sign e.g, 1-2/2 => +1-2/2
        char character = '+';
        int endIndex = 0;
        char[] words = s.toCharArray();
        while (endIndex < words.length) {

            // from previous sign -> next sign
            // get number
            if (Character.isDigit(words[endIndex])) {
                int num = 0;
                while (endIndex < words.length && Character.isDigit(words[endIndex])) {
                    num = num * 10 + Character.getNumericValue(words[endIndex]);
                    endIndex++;
                }
                stack.push(num);
            }
            // number need sign
            // character mean prefix sign
            if (endIndex == words.length || words[endIndex] != ' ') {
                if (character == '+' || character == '-') {
                    sign = character == '+' ? 1 : -1;
                    stack.push(sign * stack.pop());
                } else if (character == '*') {
                    Integer pop = stack.pop();
                    stack.push(pop * stack.pop());
                } else if (character == '/') {
                    Integer second = stack.pop();
                    Integer first = stack.pop();
                    stack.push(first / second);
                }
                if (endIndex != words.length) {
                    character = words[endIndex];
                }
            }
            endIndex++;
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
     * @param num
     * @return
     */
    public boolean isUgly(int num) {
        if (num <= 1) {
            return false;
        }
        while (true) {
            if (num == 2 || num == 3 || num == 5) {
                return true;
            }
            if (num % 5 == 0) {
                num /= 5;
            } else if (num % 3 == 0) {
                num /= 3;
            } else if (num % 2 == 0) {
                num /= 2;
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
        if (n < 1) {
            return 0;
        }
        int idx2 = 0, idx3 = 0, idx5 = 0, i = 1;
        int[] ans = new int[n];
        ans[0] = 1;
        while (i < n) {
            int tmp = Math.min(Math.min(ans[idx2] * 2, ans[idx3] * 3), ans[idx5] * 5);
            if (tmp == ans[idx2] * 2) {
                idx2++;
            } else if (tmp == ans[idx3] * 3) {
                idx3++;
            } else if (tmp == ans[idx5] * 5) {
                idx5++;
            }
            ans[i++] = tmp;

        }
        return ans[n - 1];

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


}
