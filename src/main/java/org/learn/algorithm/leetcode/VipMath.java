package org.learn.algorithm.leetcode;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class VipMath {

    public static void main(String[] args) {
        VipMath vipMath = new VipMath();
        vipMath.getHint("1807", "7810");
    }

    // 曼哈顿距离公式 (p1, p2) = |p2.x - p1.x| + |p2.y - p1.y|

    /**
     * @param grid: a 2D grid
     * @return: the minimize travel distance
     */
    public int minTotalDistance(int[][] grid) {
        // Write your code here
        if (grid == null || grid.length == 0) {
            return 0;
        }
        List<Integer> row = new ArrayList<>();
        List<Integer> columns = new ArrayList<>();
        for (int[] tmp : grid) {
            row.add(tmp[0]);
            columns.add(tmp[1]);
        }
        Collections.sort(row);
        Collections.sort(columns);
        int start = 0;
        int end = row.size() - 1;
        int result = 0;
        while (start < end) {
            result += row.get(end) - row.get(start) + columns.get(end) - columns.get(start);
            start++;
            end--;

        }
        return result;

    }


    /**
     * @param nums: A list of integers
     * @return: nothing
     */
    public void wiggleSort(int[] nums) {
        // write your code here
        if (nums == null || nums.length == 0) {
            return;
        }
        Arrays.sort(nums);
        for (int i = 1; i < nums.length; i = i + 2) {
            if (i + 1 < nums.length) {
                swap(nums, i, i + 1);
            }
        }
    }

    private void swap(int[] nums, int i, int j) {
        int val = nums[j];
        nums[j] = nums[i];
        nums[i] = val;
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
        int[] hash = new int[10];
        for (int i = 0; i < len; i++) {
            if (secret.charAt(i) == guess.charAt(i)) {
                bulls++;
            } else {
                if (hash[Character.getNumericValue(guess.charAt(i))]-- > 0) {
                    cows++;
                }
                if (hash[Character.getNumericValue(secret.charAt(i))]++ < 0) {
                    cows++;
                }

            }
        }
        return bulls + "A" + cows + "B";
    }

}
