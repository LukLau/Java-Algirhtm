package org.learn.algorithm.datastructure;

import ch.qos.logback.classic.boolex.OnErrorEvaluator;

import java.util.Iterator;

/**
 * 284. Peeking Iterator
 *
 * @author luk
 * @date 2021/4/20
 */
public class PeekingIterator implements Iterator<Integer> {
    private final Iterator<Integer> iterator;

    private Integer peek;

    public PeekingIterator(Iterator<Integer> iterator) {
        // initialize any member here.
        this.iterator = iterator;
        if (iterator.hasNext()) {
            peek = iterator.next();
        }
    }


    public Integer peek() {
        return peek;
    }

    // hasNext() and next() should behave the same as in the Iterator interface.
    // Override them if needed.
    @Override
    public Integer next() {
        Integer tmp = peek;
        if (iterator.hasNext()) {
            peek = iterator.next();
        } else {
            peek = null;
        }
        return tmp;
    }

    @Override
    public boolean hasNext() {
        return peek != null;
    }
}
