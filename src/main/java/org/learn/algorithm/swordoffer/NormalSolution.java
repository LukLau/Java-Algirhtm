package org.learn.algorithm.swordoffer;

import com.fasterxml.jackson.core.async.ByteArrayFeeder;
import org.learn.algorithm.datastructure.ListNode;
import org.learn.algorithm.datastructure.TreeNode;

import javax.swing.plaf.metal.MetalTheme;
import java.lang.annotation.ElementType;
import java.util.HashMap;
import java.util.Map;
import java.util.Stack;

/**
 * @author dora
 * @date 2021/6/10
 */
public class NormalSolution {

    public ListNode ReverseList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode reverseList = ReverseList(head.next);

        head.next.next = head;

        head.next = null;

        return reverseList;
    }

    /**
     * todo
     * lru design
     *
     * @param operators int整型二维数组 the ops
     * @param k         int整型 the k
     * @return int整型一维数组
     */
    public int[] LRU(int[][] operators, int k) {
        // write code here
        if (operators == null || operators.length == 0) {
            return new int[]{-1};
        }
        return null;
    }

    public boolean hasCycle(ListNode head) {
        if (head == null || head.next == null) {
            return false;
        }
        ListNode fast = head;
        ListNode slow = head;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (fast == slow) {
                return true;
            }
        }
        return false;
    }


    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * <p>
     * 如果目标值存在返回下标，否则返回 -1
     *
     * @param nums   int整型一维数组
     * @param target int整型
     * @return int整型
     */
    public int search(int[] nums, int target) {
        // write code here
        if (nums == null || nums.length == 0) {
            return -1;
        }
        int left = 0;
        int right = nums.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return nums[left] == target ? left : -1;
    }

    public int jumpFloor(int target) {
        if (target <= 2) {
            return target;
        }
        return jumpFloor(target - 1) + jumpFloor(target - 2);
    }


    public int maxLength(int[] arr) {
        // write code here
        if (arr == null || arr.length == 0) {
            return 0;
        }
        int left = 0;
        int result = 0;
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < arr.length; i++) {
            if (map.containsKey(arr[i])) {
                left = Math.max(left, map.get(arr[i]) + 1);
            }
            result = Math.max(result, i - left + 1);
            map.put(arr[i], i);
        }
        return result;
    }

    public int maxLengthV2(int[] arr) {
        if (arr == null || arr.length == 0) {
            return 0;
        }
        int[] hash = new int[10];
        int result = 0;
        int left = 0;
        for (int i = 0; i < arr.length; i++) {
            int val = arr[i];

            left = Math.max(left, hash[val]);

            result = Math.max(result, i - left + 1);

            hash[val] = i + 1;
        }
        return result;
    }

}
