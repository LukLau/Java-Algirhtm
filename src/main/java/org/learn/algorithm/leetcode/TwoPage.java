package org.learn.algorithm.leetcode;

import org.learn.algorithm.datastructure.ListNode;
import sun.security.util.Length;

import java.util.*;

/**
 * 第二页
 *
 * @author luk
 * @date 2021/4/12
 */
public class TwoPage {


    /**
     * 151. Reverse Words in a String
     *
     * @param s
     * @return
     */
    public String reverseWords(String s) {
        if (s == null || s.isEmpty()) {
            return "";
        }
        s = s.trim();

        String[] words = s.split(" ");
        StringBuilder builder = new StringBuilder();
        for (int i = words.length - 1; i >= 0; i--) {
            String word = words[i];
            if (word.isEmpty()) {
                continue;
            }
            builder.append(word);
            if (i != 0) {
                builder.append(" ");
            }
        }
        return builder.toString();
    }


    /**
     * 160. Intersection of Two Linked Lists
     *
     * @param headA
     * @param headB
     * @return
     */
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode p1 = headA;
        ListNode p2 = headB;
        while (p1 != p2) {
            p1 = p1 == null ? headB : p1.next;
            p2 = p2 == null ? headA : p2.next;
        }
        return p1;
    }


    /**
     * 165. Compare Version Numbers
     *
     * @param version1
     * @param version2
     * @return
     */
    public int compareVersion(String version1, String version2) {
        if (version1 == null || version2 == null) {
            return 0;
        }
        String[] words1 = version1.split("\\.");
        String[] words2 = version2.split("\\.");
        int index1 = 0;
        int index2 = 0;
        while (index1 < words1.length || index2 < words2.length) {
            Integer val1 = index1 == words1.length ? 0 : Integer.parseInt(words1[index1++]);
            Integer val2 = index2 == words2.length ? 0 : Integer.parseInt(words2[index2++]);
            if (!val1.equals(val2)) {
                return val1.compareTo(val2);
            }
        }
        return 0;
    }


    /**
     * todo
     * 166. Fraction to Recurring Decimal
     *
     * @param numerator
     * @param denominator
     * @return
     */
    public String fractionToDecimal(int numerator, int denominator) {
        return null;
    }


    /**
     * 168. Excel Sheet Column Title
     *
     * @param columnNumber
     * @return
     */
    public String convertToTitle(int columnNumber) {
        StringBuilder builder = new StringBuilder();
        while (columnNumber != 0) {
            int val = (columnNumber - 1) % 26;
            String tmp = String.valueOf((char) ('A' + val));
            builder.append(tmp);
            columnNumber = (columnNumber - 1) / 26;
        }
        return builder.reverse().toString();
    }


    public int titleToNumber(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        }
        int result = 0;
        int len = s.length();
        for (int i = 0; i < len; i++) {
            result = result * 26 + (s.charAt(i) - 'A' + 1);
        }
        return result;
    }


    public String largestNumber(int[] nums) {
        if (nums == null || nums.length == 0) {
            return "";
        }
        String[] result = new String[nums.length];
        for (int i = 0; i < nums.length; i++) {
            result[i] = String.valueOf(nums[i]);
        }
        Arrays.sort(result, new Comparator<String>() {

            @Override
            public int compare(String o1, String o2) {
                String s1 = o1 + o2;
                String s2 = o2 + o1;
                return s2.compareTo(s1);
            }
        });
        StringBuilder builder = new StringBuilder();

        if (result[0].equals("0")) {
            return "0";
        }
        for (String s : result) {
            builder.append(s);
        }
        return builder.toString();


    }


    /**
     * todo
     * 187. Repeated DNA Sequences
     *
     * @param s
     * @return
     */
    public List<String> findRepeatedDnaSequences(String s) {
        if (s == null || s.isEmpty()) {
            return new ArrayList<>();
        }
        Set<String> repeat = new HashSet<>();
        Set<String> result = new HashSet<>();
        int length = s.length();

        for (int i = 0; i < length - 9; i = i + 10) {
            String substring = s.substring(i, i + 10);
            if (repeat.add(substring)) {
                result.add(substring);
            }
        }
        return new ArrayList<>(result);
    }


    /**
     * 189. Rotate Array
     *
     * @param nums
     * @param k
     */
    public void rotate(int[] nums, int k) {
        if (nums == null || nums.length == 0 || k <= 0) {
            return;
        }
        k %= nums.length;
        reverseArray(nums, 0, nums.length - 1);
        reverseArray(nums, 0, k - 1);
        reverseArray(nums, k, nums.length - 1);
    }

    private void reverseArray(int[] str, int start, int end) {
        if (start > end) {
            return;
        }
        for (int i = start; i <= (start + end) / 2; i++) {
            swap(str, i, start + end - i);
        }
    }

    private void swap(int[] str, int i, int j) {
        int tmp = str[i];
        str[i] = str[j];
        str[j] = tmp;
    }


    // Two Sum 系列

    /**
     * 167. Two Sum II - Input array is sorted
     *
     * @param numbers
     * @param target
     * @return
     */
    public int[] twoSum(int[] numbers, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        int[] result = new int[2];
        for (int i = 0; i < numbers.length; i++) {
            int number = numbers[i];
            if (map.containsKey(target - number)) {
                result[0] = map.get(target - number) + 1;
                result[1] = i + 1;
                return result;
            }
            map.put(number, i);
        }
        return result;
    }


    public static void main(String[] args) {
        TwoPage page = new TwoPage();
        page.findRepeatedDnaSequences("AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT");
    }


}
