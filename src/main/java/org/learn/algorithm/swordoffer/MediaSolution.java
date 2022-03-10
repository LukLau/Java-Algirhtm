package org.learn.algorithm.swordoffer;

import java.util.Comparator;
import java.util.PriorityQueue;

/**
 * WC126 数据流中的中位数
 */
public class MediaSolution {

    private PriorityQueue<Integer> bigNum = new PriorityQueue<>(Comparator.reverseOrder());

    private PriorityQueue<Integer> small = new PriorityQueue<>();

    public void Insert(Integer num) {
        bigNum.offer(num);
        small.offer(bigNum.poll());

        if (small.size() > bigNum.size()) {
            bigNum.offer(small.poll());
        }

    }

    public Double GetMedian() {
        if (bigNum.size() > small.size()) {
            return bigNum.peek() / 1.0;
        }

        return (bigNum.peek() + small.peek()) / 2.0;
    }

}
