package org.dora.algorithm;

import org.dora.algorithm.data.ListNode;

import java.util.HashMap;

/**
 * @author liulu
 */
public class OneHundred {


    /**
     * 1.Two Sum
     *
     * @param nums
     * @param target
     * @return
     */
    public int[] twoSum(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return new int[2];
        }
        int[] ans = new int[2];
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(target - nums[i])) {
                ans[0] = map.get(target - nums[i]);
                ans[1] = i;
                return ans;
            }
            map.put(nums[i], i);
        }
        return ans;
    }

    /**
     * 2. Add Two Numbers
     *
     * @param l1
     * @param l2
     * @return
     */
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        if (l1 == null && l2 == null) {
            return null;
        }
        int carry = 0;
        ListNode root = new ListNode(0);
        ListNode dummy = root;
        while (l1 != null || l2 != null || carry != 0) {
            int value = (l1 == null ? 0 : l1.val) + (l2 == null ? 0 : l2.val) + carry;
            ListNode node = new ListNode(value % 10);
            carry = value / 10;
            dummy.next = node;
            dummy = dummy.next;

            l1 = l1 == null ? null : l1.next;
            l2 = l2 == null ? null : l2.next;
        }
        return root.next;
    }

    /**
     * 3. Longest Substring Without Repeating Characters
     *
     * @param s
     * @return
     */
    public int lengthOfLongestSubstring(String s) {
        if (s == null || s.length() == 0) {
            return 0;
        }
        HashMap<Character, Integer> map = new HashMap<>();
        int left = 0, max = 0;
        for (int i = 0; i < s.length(); i++) {
            if (map.containsKey(s.charAt(i))) {
                left = Math.max(left, map.get(s.charAt(i)) + 1);
            }
            map.put(s.charAt(i), i);
            max = Math.max(max, i - left + 1);
        }
        return max;
    }


    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int m = nums1.length, n = nums2.length;
        if (n < m) {
            return findMedianSortedArrays(nums2, nums1);
        }
        int imin = 0, imax = m, max_left = 0, min_right = 0;
        while (imin <= imax) {
            int i = (imin + imax) / 2;
            int j = (m + n + 1) / 2 - i;
            if (i < m && nums1[i] < nums2[j - 1]) {
                imin = i + 1;
            } else if (i > 0 && nums1[i - 1] > nums2[j]) {
                imax = i - 1;
            } else {
                if (i == 0) {
                    max_left = nums2[j - 1];
                } else if (j == 0) {
                    max_left = nums1[i - 1];
                } else {
                    max_left = Math.max(nums1[i - 1], nums2[j - 1]);
                }
                if ((m + n) % 2 == 1) {
                    return max_left;
                }
                if (i == m) {
                    min_right = nums2[j];
                } else if (j == n) {
                    min_right = nums1[i];
                } else {
                    min_right = Math.min(nums1[i], nums2[j]);
                }
                return (max_left + min_right) / 2.0;
            }
        }
        return -1;
    }

    /**
     * 5. Longest Palindromic Substring
     *
     * @param s
     * @return
     */
    public String longestPalindrome(String s) {
        if (s == null || s.length() == 0) {
            return "";
        }
        int m = s.length(), left = 0, longest = 0;
        boolean[][] dp = new boolean[m][m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j <= i; j++) {
                if (i - j <= 2) {
                    dp[j][i] = s.charAt(j) == s.charAt(i);
                } else {
                    dp[j][i] = s.charAt(j) == s.charAt(i) && dp[j + 1][i - 1];
                }
                if (dp[j][i] && i - j + 1 > longest) {
                    left = j;
                    longest = i - j + 1;
                }
            }
        }
        if (longest > 0) {
            return s.substring(left, left + longest);
        }
        return s;
    }

    /**
     * 6. ZigZag Conversion
     *
     * @param s
     * @param numRows
     * @return
     */
    public String convert(String s, int numRows) {
        if (s == null || s.length() == 0) {
            return "";
        }
        StringBuilder[] sb = new StringBuilder[numRows];
        for (int i = 0; i < numRows; i++) {
            sb[i] = new StringBuilder();
        }
        char[] chars = s.toCharArray();
        int idx = 0;
        while (idx < chars.length) {
            for (int i = 0; i < numRows && idx < chars.length; i++) {
                sb[i].append(chars[idx++]);
            }
            for (int i = numRows - 2; i >= 1 && idx < chars.length; i--) {
                sb[i].append(chars[idx++]);
            }
        }
        for (int i = 1; i < numRows; i++) {
            sb[0].append(sb[i]);
        }
        return sb[0].toString();
    }

    /**
     * 7. Reverse Integer
     *
     * @param x
     * @return
     */
    public int reverse(int x) {
        int sum = 0;
        while (x != 0) {
            if (sum > Integer.MAX_VALUE / 10 || sum < Integer.MIN_VALUE / 10) {
                return 0;
            }
            sum = sum * 10 + x % 10;
            x /= 10;
        }
        return sum;
    }

    /**
     * 8. String to Integer (atoi)
     *
     * @param str
     * @return
     */
    public int myAtoi(String str) {
        if (str == null || str.length() == 0) {
            return 0;
        }
        str = str.trim();
        if (str.length() == 0) {
            return 0;
        }
        int sign = 1, idx = 0;
        while (str.charAt(idx) == ' ') {
            idx++;
        }
        if (str.charAt(idx) == '-' || str.charAt(idx) == '+') {
            sign = str.charAt(idx) == '-' ? -1 : 1;
            idx++;
        }
        Long sum = 0L;
        while (idx < str.length() && Character.isDigit(str.charAt(idx))) {

            int value = str.charAt(idx) - '0';
            idx++;
            sum = sum * 10 + value;
            if (sum > Integer.MAX_VALUE) {
                return sign == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
            }
        }
        return sum.intValue() * sign;
    }

    /**
     * 9. Palindrome Number
     *
     * @param x
     * @return
     */
    public boolean isPalindrome(int x) {
        if (x == 0) {
            return true;
        }
        if (x < 0 || x % 10 == 0) {
            return false;
        }
        int sum = 0;
        while (x > sum) {
            sum = sum * 10 + x % 10;
            x /= 10;
        }
        return sum == x || sum / 10 == x;
    }

    public static void main(String[] args) {
        OneHundred oneHundred = new OneHundred();
        oneHundred.isPalindrome(0);
    }


}
