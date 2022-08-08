package org.learn.algorithm.datastructure;

/**
 * 277
 * Find the Celebrity
 *
 * @author luk
 * @date 2021/4/20
 */
public class CelebritySolution {

    public static void main(String[] args) {
    }


    /**
     * @param n a party with n people
     * @return the celebrity's label or -1
     */
    public int findCelebrity(int n) {
        // Write your code here
        if (n <= 0) {
            return -1;
        }
        int candidate = 0;
        for (int i = 1; i < n; i++) {
            if (knows(candidate, i)) {
                candidate = i;
            }
        }
        for (int i = 0; i < n; i++) {
            if (i == candidate) {
                continue;
            }
            if (knows(candidate, i) || !knows(i, candidate)) {
                return -1;
            }
        }
        return candidate;
    }


    private boolean knows(int a, int b) {
        return false;
    }

}
