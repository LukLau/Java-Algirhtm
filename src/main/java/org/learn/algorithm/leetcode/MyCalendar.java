package org.learn.algorithm.leetcode;

import java.util.ArrayList;
import java.util.List;
import java.util.PriorityQueue;

public class MyCalendar {
    private final List<int[]> result;

    public MyCalendar() {
        result = new ArrayList<>();
    }

    public boolean book(int start, int end) {
        if (result.isEmpty()) {
            result.add(new int[]{start, end});
            return true;
        }
        for (int[] row : result) {
            if (start < row[1] && end > row[0]) {
                return false;
            }
        }
        result.add(new int[]{start, end});
        return true;
    }

}
