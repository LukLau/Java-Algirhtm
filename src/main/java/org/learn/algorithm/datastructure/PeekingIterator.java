package org.learn.algorithm.datastructure;

import java.util.Iterator;

/**
 * 284. Peeking Iterator
 *
 * @author luk
 * @date 2021/4/20
 */
public class PeekingIterator implements Iterator<Integer> {
    private Integer peek;

    private Iterator<Integer> iterator;

    private Integer prev;

    public PeekingIterator(Iterator<Integer> iterator) {
        // initialize any member here.
        this.iterator = iterator;

        if (iterator != null && iterator.hasNext()) {
            peek = iterator.next();
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
        Integer next = iterator.hasNext() ? iterator.next() : null;

        prev = peek;

        peek = next;

        return prev;
    }

    @Override
    public boolean hasNext() {
        return peek != null;
    }
}
