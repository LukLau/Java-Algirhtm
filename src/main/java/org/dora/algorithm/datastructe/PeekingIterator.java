package org.dora.algorithm.datastructe;

import com.sun.javafx.image.IntPixelGetter;

import java.util.Iterator;
import java.util.Objects;

/**
 * date 2024年06月16日
 * @author lu.liu2
 */
public class PeekingIterator implements Iterator<Integer> {

    private Iterator<Integer> iterator = null;

    private Integer peek = null;

    boolean reachEnd = false;

    public PeekingIterator(Iterator<Integer> iterator) {
        // initialize any member here.
        this.iterator = iterator;

        if (this.hasNext()) {
            peek = this.iterator.next();
        }

    }

    // Returns the next element in the iteration without advancing the iterator.
    public Integer peek() {
        return peek;
    }

    // hasNext() and next() should behave the same as in the Iterator interface.
    // Override them if needed.
    @Override
    public Integer next() {
        Integer currentPeek = peek;

        if (currentPeek != null) {
            if (iterator.hasNext()) {
                peek = iterator.next();
            } else {
                reachEnd = true;
            }
            return currentPeek;
        }
        return peek();
    }

    @Override
    public boolean hasNext() {
        if (iterator != null && iterator.hasNext()) {
            return true;
        }
        return !reachEnd;
    }
}
