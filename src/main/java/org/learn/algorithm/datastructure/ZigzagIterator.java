package org.learn.algorithm.datastructure;

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


    private final Iterator<Integer> iterator1;

    private final Iterator<Integer> iterator2;

    boolean leftToRight = true;

    /**
     * @param v1: A 1d vector
     * @param v2: A 1d vector
     */
    public ZigzagIterator(List<Integer> v1, List<Integer> v2) {
        iterator1 = v1.iterator();
        iterator2 = v2.iterator();
        // do intialization if necessary
    }

    /**
     * @return: An integer
     */
    public int next() {
        if (!iterator1.hasNext()) {
            return iterator2.next();
        }
        if (!iterator2.hasNext()) {
            return iterator1.next();
        }
        Integer val = null;
        if (leftToRight) {
            val = iterator1.next();
        } else {
            val = iterator2.next();
        }
        leftToRight = !leftToRight;
        return val;


        // write your code here
    }

    /**
     * @return: True if has next
     */
    public boolean hasNext() {
        return iterator1.hasNext() || iterator2.hasNext();
        // write your code here
    }

}
