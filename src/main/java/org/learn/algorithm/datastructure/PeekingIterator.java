package org.learn.algorithm.datastructure;

import java.util.Iterator;

/**
 * 284. Peeking Iterator
 *
 * @author luk
 * @date 2021/4/20
 */
public class PeekingIterator implements Iterator<Integer> {
    private Integer peek = null;

    private Iterator<Integer> iterator;

    public PeekingIterator(Iterator<Integer> iterator) {
        // initialize any member here.
        this.iterator = iterator;
        if (iterator != null && iterator.hasNext()) {
            peek = iterator.next();
        }

    }

    // Returns the next element in the iteration without advancing the iterator.
    public Integer peek() {
        return this.peek;
    }

    // hasNext() and next() should behave the same as in the Iterator interface.
    // Override them if needed.
    @Override
    public Integer next() {
        Integer next = !iterator.hasNext() ? null : iterator.next();
        Integer tmp = this.peek;
        this.peek = next;
        return tmp;
    }

    @Override
    public boolean hasNext() {
        return this.peek != null;
    }


}
