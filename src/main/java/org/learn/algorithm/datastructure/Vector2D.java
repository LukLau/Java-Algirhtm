package org.learn.algorithm.datastructure;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * 251
 * Flatten 2D Vector
 *
 * @author luk
 * @date 2021/4/18
 */
public class Vector2D implements Iterable<Integer> {
    private Iterator<List<Integer>> listIterator;

    private Iterator<Integer> iterator;

    public Vector2D(List<List<Integer>> vec2d) {
        // Initialize your data structure here
        if (vec2d != null) {
            listIterator = vec2d.iterator();
        }
    }

    public Integer next() {
        // Write your code here
        if (listIterator == null) {
            return null;
        }
        if (iterator != null && iterator.hasNext()) {
            return iterator.next();
        }
        while (listIterator.hasNext()) {
            List<Integer> tmp = listIterator.next();
            if (tmp == null || tmp.isEmpty()) {
                continue;
            }
            iterator = tmp.iterator();
            break;
        }
        if (iterator != null && iterator.hasNext()) {
            return iterator.next();
        }
        return null;
    }

    //    @Override
    public boolean hasNext() {
        // Write your code here
        if (listIterator == null) {
            return false;
        }
        if (iterator != null && iterator.hasNext()) {
            return true;
        }
        if (!listIterator.hasNext()) {
            return false;
        }
        while (listIterator.hasNext()) {
            List<Integer> tmp = listIterator.next();
            if (tmp == null || tmp.isEmpty()) {
                continue;
            }
            iterator = tmp.iterator();
            break;
        }
        return iterator != null && iterator.hasNext();
    }

    //    @Override
    public void remove() {
    }

    //    @Override
    public Iterator<Integer> iterator() {
        return null;
    }
}
