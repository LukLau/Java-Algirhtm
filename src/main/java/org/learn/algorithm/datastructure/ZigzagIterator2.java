package org.learn.algorithm.datastructure;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * @author luk
 * @date 2021/8/11
 */
public class ZigzagIterator2 {

    private List<Iterator<Integer>> v1 = new ArrayList<>();

    private int iterator;


    /**
     * @param vecs: a list of 1d vectors
     */
    public ZigzagIterator2(List<List<Integer>> vecs) {
        // do intialization if necessary
        for (List<Integer> vec : vecs) {
            if (!vec.isEmpty()) {
                v1.add(vec.iterator());
            }
        }
    }

    /**
     * @return: An integer
     */
    public int next() {
        // write your code here
        iterator %= v1.size();
        Iterator<Integer> tmp = v1.get(iterator);
        Integer next = tmp.next();
        if (!tmp.hasNext()) {
            v1.remove(iterator);
            return next;
        }
        iterator++;
        return next;
    }

    /**
     * @return: True if has next
     */
    public boolean hasNext() {
        // write your code here
        for (Iterator<Integer> tmp : v1) {
            if (tmp.hasNext()) {
                return true;
            }
        }
        return false;
    }

}
