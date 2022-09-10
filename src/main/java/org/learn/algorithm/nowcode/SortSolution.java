package org.learn.algorithm.nowcode;

import java.util.*;

/**
 * @author luk
 */
public class SortSolution {

    public static void main(String[] args) {
        SortSolution solution = new SortSolution();
        Random random = new Random();
        int[] nums = new int[random.nextInt(10) + 10];
        for (int i = 0; i < nums.length; i++) {
            nums[i] = random.nextInt(50);
        }
//        solution.heapSort(nums);
//        solution.bucketSort(nums);
//        solution.countSort(nums);
//        for (int num : nums) {
//            System.out.println(num);
//        }

        nums = new int[]{0};
        solution.hIndex(nums);

    }

    private void quickSortAlgorithm(int[] nums) {
        quickSort(nums, 0, nums.length - 1);
    }

    private void mergeSortAlgorithm(int[] nums) {
        mergeSort(nums, 0, nums.length - 1);
    }

    private void quickSort(int[] nums, int start, int end) {
        if (start > end) {
            return;
        }
        int partition = partition(nums, start, end);
        quickSort(nums, start, partition - 1);
        quickSort(nums, partition + 1, end);
    }

    private int partition(int[] nums, int start, int end) {
        int pivot = nums[start];
        while (start < end) {
            while (start < end && nums[end] >= pivot) {
                end--;
            }
            if (start < end) {
                nums[start] = nums[end];
                start++;
            }
            while (start < end && nums[start] <= pivot) {
                start++;
            }
            if (start < end) {
                nums[end] = nums[start];
                end--;
            }
        }
        nums[start] = pivot;
        return start;
    }


    private void mergeSort(int[] nums, int start, int end) {
        if (start >= end) {
            return;
        }
        int mid = start + (end - start) / 2;
        mergeSort(nums, start, mid);
        mergeSort(nums, mid + 1, end);
        merge(nums, start, mid, end);
    }

    private void merge(int[] nums, int start, int mid, int end) {
        int[] result = new int[end - start + 1];
        int i = start;
        int k = mid + 1;
        int index = 0;
        while (i <= mid && k <= end) {
            if (nums[i] < nums[k]) {
                result[index++] = nums[i++];
            } else {
                result[index++] = nums[k++];
            }
        }
        while (i <= mid) {
            result[index++] = nums[i++];
        }
        while (k <= end) {
            result[index++] = nums[k++];
        }
        System.arraycopy(result, 0, nums, start, result.length);
    }

    private void heapSort(int[] nums) {
        if (nums == null || nums.length == 0) {
            return;
        }
        for (int i = nums.length / 2 - 1; i >= 0; i--) {
            adjustHeapSort(nums, i, nums.length);
        }
        for (int i = nums.length - 1; i > 0; i--) {
            swap(nums, 0, i);
            adjustHeapSort(nums, 0, i);
        }

    }

    private void swap(int[] nums, int i, int j) {
        int tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;
    }


    private void adjustHeapSort(int[] nums, int i, int len) {
        int pivot = nums[i];
        for (int k = 2 * i + 1; k < len; k = 2 * k + 1) {
            if (k + 1 < len && nums[k] < nums[k + 1]) {
                k = k + 1;
            }
            if (nums[k] > pivot) {
                nums[i] = nums[k];
                i = k;
            } else {
                break;
            }
        }
        nums[i] = pivot;
    }


    public void bucketSort(int[] nums) {
        if (nums == null || nums.length == 0) {
            return;
        }
        int max = Integer.MIN_VALUE;
        int min = Integer.MAX_VALUE;
        for (int num : nums) {
            if (num > max) {
                max = num;
            }
            if (num < min) {
                min = num;
            }
        }
        int bucketSide = (max - min) / nums.length + 1;
        int bucketCounts = (max - min) / bucketSide + 1;

        List<List<Integer>> buckets = new ArrayList<>();

        for (int i = 0; i < bucketCounts; i++) {
            List<Integer> tmp = new ArrayList<>();
            buckets.add(tmp);
        }
        for (int num : nums) {
            int index = (num - min) / bucketSide;

            buckets.get(index).add(num);
        }
        int index = 0;
        for (int i = 0; i < bucketCounts; i++) {
            List<Integer> currentBucket = buckets.get(i);

            if (currentBucket != null && !currentBucket.isEmpty()) {
                Collections.sort(currentBucket);
            }
            assert currentBucket != null;
            for (Integer currentItem : currentBucket) {
                nums[index++] = currentItem;
            }
        }
    }

    public void countSort(int[] nums) {
        if (nums == null || nums.length == 0) {
            return;
        }
        int min = Integer.MAX_VALUE;
        int max = Integer.MIN_VALUE;
        for (int num : nums) {
            if (num < min) {
                min = num;
            }
            if (num > max) {
                max = num;
            }
        }
        int[] tmp = new int[max - min + 1];
        for (int num : nums) {
            tmp[num - min]++;
        }
        int[] result = new int[nums.length];
        int index = 0;
        for (int i = 0; i < tmp.length; i++) {
            int count = tmp[i];
            while (count-- > 0) {
                result[index++] = min + i;
            }
        }
        System.arraycopy(result, 0, nums, 0, result.length);
    }


    /**
     * https://leetcode.com/problems/h-index/
     * <p>
     * https://leetcode.com/problems/h-index-ii/discuss/71063/Standard-binary-search
     *
     * @param citations
     * @return
     */
    public int hIndex(int[] citations) {
        int[] count = new int[citations.length + 1];
        for (int citation : citations) {
            if (citation >= citations.length) {
                count[citations.length]++;
            } else {
                count[citation]++;
            }
        }
        int total = 0;
        for (int i = count.length - 1; i >= 0; i--) {
            total += count[i];
            if (total >= i) {
                return i;
            }
        }
        return 0;


    }


}
