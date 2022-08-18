package org.learn.algorithm.datastructure;

import java.util.Comparator;
import java.util.PriorityQueue;

/**
 * 295. Find Median from Data Stream
 *
 * @author luk
 * @date 2021/4/24
 */
public class MedianFinder {

    public static void main(String[] args) {
        MedianFinder medianFinder = new MedianFinder();
        medianFinder.addNum(1);
        medianFinder.addNum(2);
        medianFinder.addNum(3);

        double median = medianFinder.findMedian();

        System.out.println(median);
//        medianFinder.addNum(4);
//        medianFinder.addNum(5);
//        medianFinder.addNum(6);


    }

    private PriorityQueue<Integer> previous = new PriorityQueue<>(Comparator.reverseOrder());

    private PriorityQueue<Integer> after = new PriorityQueue<>();

    public MedianFinder() {

    }

    public void addNum(int num) {
        previous.offer(num);
        after.offer(previous.poll());
        if (after.size() > previous.size()) {
            previous.offer(after.poll());
        }


    }

    public double findMedian() {
        if (previous.size() > after.size()) {
            return previous.peek() / 1.0;
        }
        return (previous.peek() + after.peek()) / 2.0;
    }


}
