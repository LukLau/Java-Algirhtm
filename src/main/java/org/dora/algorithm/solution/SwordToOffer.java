package org.dora.algorithm.solution;

/**
 * @author liulu
 * @date 2019-03-21
 */
public class SwordToOffer {
    public static void main(String[] args) {
        SwordToOffer swordToOffer = new SwordToOffer();
        int[] numbers = new int[]{1, 0, 0, 1, 0};
        swordToOffer.isContinuous(numbers);
    }

    /**
     * 扑克牌顺子
     *
     * @param numbers
     * @return
     */
    public boolean isContinuous(int[] numbers) {
        if (numbers == null || numbers.length == 0) {
            return false;
        }
        int[] hash = new int[26];
        int min = 13;
        int max = 0;
        int countOfZero = 0;
        for (int value : numbers) {
            if (value == 0) {
                countOfZero++;
                continue;
            }
            if (hash[value] != 0) {
                return false;
            }
            if (value > max) {
                max = value;
            }
            if (value < min) {
                min = value;
            }
            hash[value]++;
        }
        if (countOfZero >= 5) {
            return true;
        }
        if (max - min > 4) {
            return false;
        }
        return true;

    }
}
