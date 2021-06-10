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
    private Iterator<List<Integer>> iterator1;

    private Iterator<Integer> iterator2;

    public Vector2D(List<List<Integer>> vec2d) {
        iterator1 = vec2d.iterator();
        // Initialize your data structure here
    }

    public Integer next() {
        if (iterator2 != null && iterator2.hasNext()) {
            return iterator2.next();
        }
        while (iterator1.hasNext()) {
            List<Integer> next = iterator1.next();
            iterator2 = next.iterator();
            if (iterator2.hasNext()) {
                return iterator2.next();
            }
        }
        return null;
        // Write your code here
    }

    public boolean hasNext() {
        // Write your code here
        if (iterator2 != null && iterator2.hasNext()) {
            return true;
        }
        while (iterator1.hasNext()) {
            List<Integer> next = iterator1.next();
            iterator2 = next.iterator();
            if (iterator2.hasNext()) {
                return true;
            }
        }
        return false;
    }

    public void remove() {
    }

    @Override
    public Iterator<Integer> iterator() {
        return null;
    }

    public static void main(String[] args) {
        Vector2D vector2D = new Vector2D(new ArrayList<>());
        if (vector2D.hasNext()) {
            vector2D.next();
        }
    }
}
