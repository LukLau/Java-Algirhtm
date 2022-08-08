package org.learn.algorithm.leetcode;

public class VipMath {

    public static void main(String[] args) {
        VipMath vipMath = new VipMath();
        vipMath.getHint("1807", "7810");
    }


    /**
     * 263. Ugly Number
     *
     * @param n
     * @return
     */
    public boolean isUgly(int n) {
        if (n < 1) {
            return false;
        }
        if (n == 1) {
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
            int value = Math.min(result[index2] * 2, Math.min(result[index3] * 3, result[index5] * 5));
            if (value == result[index2] * 2) {
                index2++;
            }
            if (value == result[index3] * 3) {
                index3++;
            }
            if (value == result[index5] * 5) {
                index5++;
            }
            result[index] = value;
            index++;
        }
        return result[n - 1];

    }


    /**
     * 299. Bulls and Cows
     *
     * @param secret
     * @param guess
     * @return
     */
    public String getHint(String secret, String guess) {
        if (secret == null || guess == null) {
            return "";
        }
        int bulls = 0;
        int cows = 0;
        int len = secret.length();
        int[] hash1 = new int[10];
        for (int i = 0; i < len; i++) {
            if (secret.charAt(i) == guess.charAt(i)) {
                bulls++;
            } else {
                if (hash1[Character.getNumericValue(guess.charAt(i))]-- > 0) {
                    cows++;
                }
                if (hash1[Character.getNumericValue(secret.charAt(i))]++ < 0) {
                    cows++;
                }
            }
        }
        return bulls + "A" + cows + "B";
    }
}
