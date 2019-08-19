package org.dora.algorithm.solution.v1;

/**
 * @author liulu
 * @date 2019-03-14
 */
public class SortSolution {
    public static void main(String[] args) {
        int[] arrays = new int[]{1, 3, 9, 7, 6, -1, -100, 200, 5};
        SortSolution sortSolution = new SortSolution();
//        sortSolution.bubbleSort(arrays);
//        sortSolution.quickSort(arrays, 0, arrays.length - 1);
        sortSolution.heapSort(arrays);
        for (int num : arrays) {
            System.out.println(num);
        }
    }

    private void bubbleSort(int[] arrays) {
        if (arrays == null || arrays.length == 0) {
            return;
        }
        boolean needSort = true;
        for (int i = 0; i < arrays.length && needSort; i++) {
            needSort = false;
            for (int j = arrays.length - 1; j > i; j--) {
                if (arrays[j] < arrays[j - 1]) {
                    needSort = !needSort;
                    this.swap(arrays, j, j - 1);
                }
            }
        }
    }

    private void swap(int[] arrays, int start, int end) {
        int tmp = arrays[start];
        arrays[start] = arrays[end];
        arrays[end] = tmp;
    }

    private void quickSort(int[] arrays, int left, int right) {
        if (arrays == null || arrays.length == 0) {
            return;
        }
        if (left > right) {
            return;
        }
        int index = this.partition(arrays, left, right);
        this.quickSort(arrays, left, index - 1);
        this.quickSort(arrays, index + 1, right);
    }

    private int partition(int[] nums, int left, int right) {
        int pivot = nums[left];
        while (left < right) {
            while (left < right && nums[right] >= pivot) {
                right--;
            }
            if (left < right) {
                nums[left] = nums[right];
                left++;
            }
            while (left < right && nums[left] < pivot) {
                left++;
            }
            if (left < right) {
                nums[right] = nums[left];
                right--;
            }
        }
        nums[left] = pivot;
        return left;
    }

    private void heapSort(int[] nums) {
        if (nums == null || nums.length == 0) {
            return;
        }
        for (int i = nums.length / 2 - 1; i >= 0; i--) {
            this.adjustHeap(nums, i, nums.length);
        }
        for (int i = nums.length - 1; i > 0; i--) {
            this.swap(nums, 0, i);
            this.adjustHeap(nums,0, i);
        }
    }

    private void adjustHeap(int[] nums, int i, int len) {
        int tmp = nums[i];
        for (int k = 2 * i + 1; k < len; k = 2 * k + 1) {
            if (k + 1 < len && nums[k] < nums[k+1]) {
                    k++;
            }
            if (nums[k] > tmp) {
                this.swap(nums, i, k);
                i = k;
            } else {
                break;
            }
        }

    }
}
