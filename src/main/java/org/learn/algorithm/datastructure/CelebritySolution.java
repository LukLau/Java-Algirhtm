package org.learn.algorithm.datastructure;

/**
 * 277
 * Find the Celebrity
 *
 * @author luk
 * @date 2021/4/20
 */
public class CelebritySolution {


    /**
     * @param n a party with n people
     * @return the celebrity's label or -1
     */
    public int findCelebrity(int n) {
        int candidate = 0;
        for (int i = 1; i < n; i++) {
            if (!knows(i, candidate)) {
                candidate = i;
            }
        }
        for (int i = 0; i < n; i++) {
            if (i != candidate) {
                if (!knows(i, candidate) || knows(candidate, i)) {
                    return -1;
                }
            }
        }
        return candidate;
    }

    private boolean knows(int a, int b) {
        return false;
    }

}
