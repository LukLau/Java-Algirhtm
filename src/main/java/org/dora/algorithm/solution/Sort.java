package org.dora.algorithm.solution;

import jdk.nashorn.internal.runtime.FindProperty;

import java.util.Arrays;
import java.util.Random;

/**
 * date 2024年04月25日
 */
public class Sort {

    public static void main(String[] args) {
        Sort sort = new Sort();
        int[] params = new int[10];
        Random random = new Random();
        for (int i = 0; i < 10; i++) {
//            int[] nums = new int[]{3, 4, 4, 6, 2};
            params[i] = random.nextInt(100);
        }
//        sort.heapSort(nums);
        sort.quickSort(params);

        for (int num : params) {
            System.out.println(num);
        }
    }

    public void bubbleSort(int[] nums) {
        if (nums == null || nums.length == 0) {
            return;
        }
        for (int i = 0; i < nums.length - 1; i++) {
            for (int j = nums.length - 1; j > i; j--) {
                if (nums[j] < nums[j - 1]) {
                    swap(nums, j, j - 1);
                }
            }
        }
    }

    public void quickSort(int[] nums) {
        if (nums == null || nums.length == 0) {
            return;
        }
        internalQuickSort(nums, 0, nums.length - 1);
    }

    private int getPartition(int[] nums, int start, int end) {
        int pivot = nums[start];
        while (start < end) {
            while (start < end && nums[end] >= pivot) {
                end--;
            }
            if (start < end) {
                swap(nums, start, end);
                start++;
            }
            while (start < end && nums[start] <= pivot) {
                start++;
            }
            if (start < end) {
                swap(nums, start, end);
                end--;
            }
        }
        nums[start] = pivot;

        return start;
    }

    private void internalQuickSort(int[] nums, int start, int end) {
        if (start >= end) {
            return;
        }
        int partition = getPartition(nums, start, end);
        internalQuickSort(nums, start, partition - 1);
        internalQuickSort(nums, partition + 1, end);
    }

//    public void mergeSort(int[] nums) {
//        if (nums == null || nums.length == 0) {
//            return;
//        }
//        internalMergeSort(nums, 0, nums.length - 1);
//    }
//
//    private void internalMergeSort(int[] nums, int start, int end) {
//        if (start >= end) {
//            return;
//        }
//        int mid = start + (end - start) / 2;
//        internalMergeSort(nums, start, mid);
//        internalMergeSort(nums, mid + 1, end);
//        internalMerge(nums, start, mid, end);
//    }
//
//    private void internalMerge(int[] nums, int start, int mid, int end) {
//        int i = start;
//        int j = mid + 1;
//        int index = 0;
//        int[] tmp = new int[end - start + 1];
//
//        while (i <= mid && j <= end) {
//            if (nums[i] <= nums[j]) {
//                tmp[index++] = nums[i++];
//            } else {
//                tmp[index++] = nums[j++];
//            }
//        }
//        while (i <= mid) {
//            tmp[index++] = nums[i++];
//        }
//        while (j <= end) {
//            tmp[index++] = nums[j++];
//        }
//        System.arraycopy(tmp, 0, nums, start, tmp.length);
//    }
//
//    public void heapSort(int[] nums) {
//        if (nums == null || nums.length == 0) {
//            return;
//        }
//        for (int i = nums.length / 2 - 1; i >= 0; i--) {
//            adjustHeap(nums, i, nums.length);
//        }
//        for (int i = nums.length - 1; i > 0; i--) {
//            swap(nums, 0, i);//将堆顶元素与末尾元素进行交换
//            adjustHeap(nums, 0, i);//重新对堆进行调整
//        }
//
//    }
//
//    private void adjustHeap(int[] nums, int start, int end) {
//
//        int tmp = nums[start];
//        for (int j = 2 * start + 1; j < end; j = 2 * j + 1) {
//            if (j + 1 < end && nums[j] < nums[j + 1]) {
//                j++;
//            }
//            if (nums[j] > tmp) {
//                nums[start] = nums[j];
//                start = j;
//            } else {
//                break;
//            }
//        }
//        nums[start] = tmp;
//
//    }

    private void swap(int[] nums, int i, int j) {
        int tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;
    }

}
