package org.learn.algorithm.swordoffer;


import java.util.Random;

/**
 * 常用排序问题
 *
 * @author luk
 * @date 2021/5/11
 */
public class SortSolution {

    public static void main(String[] args) {
        SortSolution sortSolution = new SortSolution();
        int n = 10;
        int[] nums = new int[n];
        Random random = new Random();
        for (int i = 0; i < n; i++) {
            nums[i] = random.nextInt(100);
        }
//        sortSolution.heapSort(nums, 0, nums.length - 1);
        sortSolution.heapSort(nums);
        for (int num : nums) {
            System.out.println(num + " ");
        }
    }

    /**
     * 快速排序
     */
    public void quickSort(int[] nums, int start, int end) {
        if (start > end) {
            return;
        }
        int partition = partition(nums, start, end);
        quickSort(nums, start, partition - 1);
        quickSort(nums, partition + 1, end);
    }

    public int partition(int[] nums, int low, int high) {
        int pivot = nums[low];
        while (low < high) {
            while (low < high && nums[high] >= pivot) {
                high--;
            }
            if (low < high) {
                swap(nums, low, high);
                low++;
            }
            while (low < high && nums[low] <= pivot) {
                low++;
            }
            if (low < high) {
                swap(nums, low, high);
                high--;
            }
        }
        nums[low] = pivot;
        return low;
    }

    private void swap(int[] nums, int i, int j) {
        int val = nums[i];
        nums[i] = nums[j];
        nums[j] = val;
    }


    /**
     * 归并排序
     */

    public void mergeSort(int[] nums, int start, int end) {
        if (start >= end) {
            return;
        }
        int mid = start + (end - start) / 2;
        mergeSort(nums, start, mid);
        mergeSort(nums, mid + 1, end);
        merge(nums, start, mid, end);
    }

    private void merge(int[] nums, int start, int mid, int end) {
        int[] tmp = new int[end - start + 1];
        int left = start;
        int k = mid + 1;
        int index = 0;
        while (left <= mid && k <= end) {
            if (nums[left] <= nums[k]) {
                tmp[index++] = nums[left++];
            } else {
                tmp[index++] = nums[k++];
            }
        }
        while (left <= mid) {
            tmp[index++] = nums[left++];
        }
        while (k <= end) {
            tmp[index++] = nums[k++];
        }
        index = start;
        System.arraycopy(tmp, 0, nums, index, tmp.length);
    }

    /**
     * 堆排序
     */
    public void heapSort(int[] nums) {
        if (nums == null || nums.length == 0) {
            return;
        }
        for (int i = nums.length / 2 - 1; i >= 0; i--) {
            adjustHeap(nums, 0, i);
        }
        for (int i = nums.length - 1; i > 0; i--) {
            swap(nums, i, 0);
            adjustHeap(nums, 0, i);
        }
    }

    private void adjustHeap(int[] nums, int k, int len) {
        int tmp = nums[k];
        for (int i = 2 * k + 1; i < len; i = 2 * i + 1) {
            if (i + 1 < len && nums[i] < nums[i + 1]) {
                i = i + 1;
            }
            if (nums[i] < nums[k]) {
                break;
            }
            swap(nums, i, k);
            k = i;
        }
        nums[k] = tmp;
    }

}
