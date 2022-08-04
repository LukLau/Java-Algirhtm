package org.learn.algorithm.leetcode;

import javax.xml.transform.sax.SAXResult;

public class VipMath {

    public static void main(String[] args) {
        VipMath vipMath = new VipMath();
        vipMath.getHint("1807", "7810");
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
//        int[] hash2 = new int[10];
        for (int i = 0; i < len; i++) {
            hash1[Character.getNumericValue(secret.charAt(i))]++;
            hash1[Character.getNumericValue(guess.charAt(i))]--;
        }
        for (int i = 0; i < len; i++) {
            if (secret.charAt(i) == guess.charAt(i)) {
                bulls++;
            } else {
                if (hash1[Character.getNumericValue(guess.charAt(i))]++ > 0) {
                    cows++;
                }
                if (hash1[Character.getNumericValue(secret.charAt(i))]-- < 0) {
                    cows++;
                }
            }
        }
        return bulls + "A" + cows + "B";
    }

}
