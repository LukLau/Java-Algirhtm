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
        listIterator = vec2d.iterator();
    }

    public Integer next() {
        // Write your code here
        if (iterator != null && iterator.hasNext()) {
            return iterator.next();
        }
        while (listIterator.hasNext()) {
            iterator = listIterator.next().iterator();
            while (iterator.hasNext()) {
                Integer next = iterator.next();
                if (next != null) {
                    return next;
                }
            }
        }
        return null;
    }

    public boolean hasNext() {
        // Write your code here
        if (iterator != null && iterator.hasNext()) {
            return true;
        }
        while (listIterator.hasNext()) {
            iterator = listIterator.next().iterator();
            if (iterator.hasNext()) {
                return true;
            }
        }
        return false;
    }


    public static void main(String[] args) {
        Vector2D vector2D = new Vector2D(new ArrayList<>());
        if (vector2D.hasNext()) {
            vector2D.next();
        }
    }

    @Override
    public Iterator<Integer> iterator() {
        return null;
    }
}
