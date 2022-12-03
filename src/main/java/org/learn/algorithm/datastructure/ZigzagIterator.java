package org.learn.algorithm.datastructure;

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

/**
 * 281
 * Zigzag Iterator
 *
 * @author luk
 * @date 2021/4/20
 */
public class ZigzagIterator {
    private Iterator<Integer> iterator1;

    private Iterator<Integer> iterator2;

    private boolean leftToRight;

    /*
     * @param v1: A 1d vector
     * @param v2: A 1d vector
     */
    public ZigzagIterator(List<Integer> v1, List<Integer> v2) {
        // do intialization if necessary
        if (v1 != null) {
            iterator1 = v1.iterator();
        }
        if (v2 != null) {
            iterator2 = v2.iterator();
        }
        leftToRight = true;
    }

    /*
     * @return: An integer
     */
    public int next() {
        // write your code here
        if (iterator1 == null && iterator2 == null) {
            return -1;
        }
        if (iterator1 == null || !iterator1.hasNext()) {
            return iterator2 == null ? -1 : iterator2.next();
        }
        if (iterator2 == null || !iterator2.hasNext()) {
            return iterator1.next();
        }
        int result = -1;
        if (leftToRight) {
            result = iterator1.next();
        } else {
            result = iterator2.next();
        }
        leftToRight = !leftToRight;
        return result;
    }

    /*
     * @return: True if has next
     */
    public boolean hasNext() {
        // write your code here
        if (iterator1 == null && iterator2 == null) {
            return false;
        }
        return (iterator1 != null && iterator1.hasNext()) || (iterator2 != null && iterator2.hasNext());
    }


    public static void main(String[] args) {
        List<Integer> v1 = Arrays.asList(1, 2);
        List<Integer> v2 = Arrays.asList(3, 4, 5, 6);
        ZigzagIterator zigzagIterator = new ZigzagIterator(v1, v2);
        zigzagIterator.next();
        zigzagIterator.next();
        zigzagIterator.next();
        zigzagIterator.next();
        zigzagIterator.next();
        zigzagIterator.next();
    }

}
