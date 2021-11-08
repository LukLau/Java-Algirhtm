package org.learn.algorithm.datastructure;

/**
 * 278. First Bad Version
 *
 * @author luk
 * @date 2021/4/20
 */
public class VersionControlSolution {


    public int firstBadVersion(int n) {
        int start = 1;
        int end = n;
        while (start < end) {
            int mid = start + (end - start) / 2;
            if (isBadVersion(mid)) {
                end = mid;
            } else {
                start = mid + 1;
            }
        }
        return start;
    }

    private boolean isBadVersion(int version) {
        return false;
    }

}
