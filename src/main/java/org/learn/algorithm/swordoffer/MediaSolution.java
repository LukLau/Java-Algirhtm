package org.learn.algorithm.swordoffer;

import java.util.Comparator;
import java.util.PriorityQueue;

/**
 * WC126 数据流中的中位数
 */
public class MediaSolution {

    private final PriorityQueue<Integer> small = new PriorityQueue<>(Comparator.reverseOrder());

    private final PriorityQueue<Integer> big = new PriorityQueue<>();


    /**
     * @param val: a num from the data stream.
     * @return: nothing
     */
    public void add(int val) {
        // write your code here
        small.offer(val);
        big.offer(small.poll());

        if (big.size() > small.size()) {
            small.offer(big.poll());
        }
    }

    /**
     * @return: return the median of the all numbers
     */
    public int getMedian() {
        // write your code here
        if ((small.size() + big.size() - 1) / 2 <= small.size()) {
            return small.peek();
        }
        return big.peek();
    }

    public static void main(String[] args) {
        MediaSolution mediaSolution = new MediaSolution();

        mediaSolution.add(1);
        System.out.println(mediaSolution.getMedian());
        mediaSolution.add(2);
        System.out.println(mediaSolution.getMedian());
        mediaSolution.add(3);
        System.out.println(mediaSolution.getMedian());
        mediaSolution.add(4);
        System.out.println(mediaSolution.getMedian());
        mediaSolution.add(5);
        System.out.println(mediaSolution.getMedian());
    }


}
