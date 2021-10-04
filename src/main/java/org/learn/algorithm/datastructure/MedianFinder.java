package org.learn.algorithm.datastructure;

import java.util.PriorityQueue;

/**
 * 295. Find Median from Data Stream
 *
 * @author luk
 * @date 2021/4/24
 */
public class MedianFinder {

    private final PriorityQueue<Integer> small;

    private final PriorityQueue<Integer> big;


    /**
     * initialize your data structure here.
     */
    public MedianFinder() {
        small = new PriorityQueue<>((o1, o2) -> o2 - o1);
        big = new PriorityQueue<>();

    }

    public void addNum(int num) {
        small.offer(num);
        big.offer(small.poll());
        if (big.size() > small.size()) {
            small.offer(big.poll());
        }
    }

    public double findMedian() {
        if (small.size() > big.size()) {
            return small.peek() / 1.0;
        }
        return (small.peek() + big.peek()) / 2.0;
    }
}
