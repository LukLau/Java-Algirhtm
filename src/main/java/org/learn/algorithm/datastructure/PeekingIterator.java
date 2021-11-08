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
        peek = iterator.next();
    }

    /**
     * Returns the next element in the iteration without advancing the iterator.
     */

    public Integer peek() {
        return peek;
    }

    @Override
    public Integer next() {
        Integer next = iterator.hasNext() ? iterator.next() : null;
        int pre = peek;
        peek = next;
        return pre;
    }

    @Override
    public boolean hasNext() {
        return peek != null;
    }
}
