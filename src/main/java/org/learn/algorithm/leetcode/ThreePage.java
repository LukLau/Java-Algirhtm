package org.learn.algorithm.leetcode;

import org.learn.algorithm.datastructure.ListNode;
import org.learn.algorithm.datastructure.TreeNode;

import java.util.*;

/**
 * 第三页
 *
 * @author luk
 * @date 2021/4/13
 */
public class ThreePage {

    public static void main(String[] args) {
        ThreePage page = new ThreePage();
        int[] nums = new int[]{0, 1, 2, 4, 5, 7};
        page.summaryRanges(nums);
    }


    /**
     * 202. Happy Number
     */
    public boolean isHappy(int n) {
        List<Integer> result = new ArrayList<>();
        while (true) {
            int tmp = 0;
            while (n != 0) {
                int remain = n % 10;
                tmp += remain * remain;
                n /= 10;
            }
            if (tmp == 1) {
                return true;
            }
            if (result.contains(tmp)) {
                return false;
            }
            result.add(tmp);
            n = tmp;
        }
    }


    /**
     * 203. Remove Linked List Elements
     *
     * @param head
     * @param val
     * @return
     */
    public ListNode removeElements(ListNode head, int val) {
        if (head == null) {
            return null;
        }
        if (head.val == val) {
            return removeElements(head.next, val);
        }
        head.next = removeElements(head.next, val);
        return head;
    }


    /**
     * 205. Isomorphic Strings
     *
     * @param s
     * @param t
     * @return
     */
    public boolean isIsomorphic(String s, String t) {
        if (s == null || t == null) {
            return false;
        }
        Map<Character, Integer> map1 = new HashMap<>();
        Map<Character, Integer> map2 = new HashMap<>();
        int m = s.length();
        int n = t.length();
        if (m != n) {
            return false;
        }
        for (int i = 0; i < m; i++) {
            if (!Objects.equals(map1.put(s.charAt(i), i), map2.put(t.charAt(i), i))) {
                return false;
            }
        }
        return true;
    }


    /**
     * 206. Reverse Linked List
     *
     * @param head
     * @return
     */
    public ListNode reverseList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode prev = null;
        while (head != null) {
            ListNode tmp = head.next;
            head.next = prev;
            prev = head;
            head = tmp;
        }
        return prev;
    }


    /**
     * todo use OlogN
     * 209. Minimum Size Subarray Sum
     *
     * @param s
     * @param nums
     * @return
     */
    public int minSubArrayLen(int s, int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int result = Integer.MAX_VALUE;
        int sum = 0;
        int left = 0;
        int endIndex = 0;
        while (endIndex < nums.length) {
            sum += nums[endIndex];
            while (sum >= s) {
                result = Math.min(result, endIndex - left + 1);
                sum -= nums[left++];
            }
            endIndex++;
        }
        return result == Integer.MAX_VALUE ? 0 : result;
    }

    /**
     * 215. Kth Largest Element in an Array
     *
     * @param nums
     * @param k
     * @return
     */
    public int findKthLargest(int[] nums, int k) {
        if (nums == null || nums.length == 0) {
            return -1;
        }
        PriorityQueue<Integer> queue = new PriorityQueue<>(nums.length, Comparator.reverseOrder());
        for (int num : nums) {
            queue.offer(num);
        }
        int iterator = 1;
        while (iterator < k) {
            Integer poll = queue.poll();
            iterator++;
        }
        return queue.poll();
    }


    /**
     * 217. Contains Duplicate
     *
     * @param nums
     * @return
     */
    public boolean containsDuplicate(int[] nums) {
        if (nums == null || nums.length == 0) {
            return false;
        }
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            if (map.containsKey(num)) {
                return true;
            }
            map.put(num, 1);
        }
        return false;
    }


    /**
     * 218. The Skyline Problem
     * todo
     *
     * @param buildings
     * @return
     */
    public List<List<Integer>> getSkyline(int[][] buildings) {
        return null;
    }


    /**
     * 219. Contains Duplicate II
     *
     * @param nums
     * @param k
     * @return
     */
    public boolean containsNearbyDuplicate(int[] nums, int k) {
        Set<Integer> set = new HashSet<>();
        for (int i = 0; i < nums.length; i++) {
            if (i > k) {
                set.remove(nums[i - 1 - k]);
            }
            if (!set.add(nums[i])) {
                return true;
            }
        }
        return false;
    }


    /**
     * todo
     * 220. Contains Duplicate III
     *
     * @param nums
     * @param k
     * @param t
     * @return
     */
    public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
        return false;
    }

    /**
     * 222. Count Complete Tree Nodes
     *
     * @param root
     * @return
     */
    public int countNodes(TreeNode root) {
        if (root == null) {
            return 0;
        }
        return 1 + countNodes(root.left) + countNodes(root.right);
    }


    /**
     * todo
     * 223. Rectangle Area
     *
     * @param A
     * @param B
     * @param C
     * @param D
     * @param E
     * @param F
     * @param G
     * @param H
     * @return
     */
    public int computeArea(int A, int B, int C, int D, int E, int F, int G, int H) {

        return -1;
    }


    public TreeNode invertTree(TreeNode root) {
        if (root == null) {
            return null;
        }
        TreeNode left = root.left;
        root.left = root.right;
        root.right = left;
        invertTree(root.left);
        invertTree(root.right);
        return root;
    }


    /**
     * 228. Summary Ranges
     *
     * @param nums
     * @return
     */
    public List<String> summaryRanges(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new ArrayList<>();
        }
        List<String> result = new ArrayList<>();
        int startIndex = 0;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] != nums[i - 1] + 1) {
                result.add(range(nums[startIndex], nums[i - 1]));
                startIndex = i;
            }
        }
        if (startIndex <= nums.length - 1) {
            result.add(range(nums[startIndex], nums[nums.length - 1]));
        }
        return result;
    }

    private String range(int start, int end) {
        return start == end ? String.valueOf(start) : start + "->" + end;
    }


    /**
     * 234. Palindrome Linked List
     *
     * @param head
     * @return
     */
    public boolean isPalindrome(ListNode head) {
        if (head == null || head.next == null) {
            return false;
        }
        ListNode fast = head;
        ListNode slow = head;
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        ListNode next = slow.next;
        slow.next = null;

        ListNode reverseList = reverseList(next);
        while (head != null && reverseList != null) {
            if (head.val != reverseList.val) {
                return false;
            }
            head = head.next;
            reverseList = reverseList.next;
        }
        return true;
    }


    /**
     * 237. Delete Node in a Linked List
     *
     * @param node
     */
    public void deleteNode(ListNode node) {
        if (node.next.next == null) {
            node.val = node.next.val;
            node.next = null;
            return;
        }
        node.val = node.next.val;
        node.next = node.next.next;
    }


    /**
     * 238. Product of Array Except Self
     *
     * @param nums
     * @return
     */
    public int[] productExceptSelf(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new int[]{};
        }
        int[] result = new int[nums.length];
        int base = 1;
        for (int i = 0; i < nums.length; i++) {
            result[i] = base;
            base *= nums[i];
        }
        base = 1;
        for (int i = nums.length - 1; i >= 0; i--) {
            result[i] *= base;
            base *= nums[i];
        }
        return result;
    }


    /**
     * todo
     *
     * @param num
     * @return
     */
    public String numberToWords(int num) {
        if (num < 0) {
            return "";
        }

        return "";
    }

    private String convertToWords(int num) {
        String[] v1 = new String[]{"", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"};

        String[] v2 = new String[]{"", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"};


        int a = num / 100;

        int b = num % 100;

        int c = num % 10;

        String res = "";

//        res = b < 20 ? v1[b] : (v2[b / 10] +

        return res;

    }


    /**
     * 283. Move Zeroes
     *
     * @param nums
     */
    public void moveZeroes(int[] nums) {
        if (nums == null || nums.length == 0) {
            return;
        }
        int index = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != 0) {
                nums[index++] = nums[i];
            }
        }
        while (index < nums.length) {
            nums[index++] = 0;
        }
    }

    /**
     * 290. Word Pattern
     *
     * @param pattern
     * @param s
     * @return
     */
    public boolean wordPattern(String pattern, String s) {
        if (pattern == null || s == null) {
            return false;
        }
        String[] words = s.split(" ");
        if (pattern.length() != words.length) {
            return false;
        }
        Map<String, Integer> map2 = new HashMap<>();
        Map<Character, Integer> map1 = new HashMap<>();
        for (int i = 0; i < words.length; i++) {
            if (!Objects.equals(map1.put(pattern.charAt(i), i), map2.put(words[i], i))) {
                return false;
            }
        }
        return true;
    }


    /**
     * todo
     * 299. Bulls and Cows
     *
     * @param secret
     * @param guess
     * @return
     */
    public String getHint(String secret, String guess) {
        int bulls = 0;
        int cows = 0;
        char[] secretWords = secret.toCharArray();
        char[] guessWords = guess.toCharArray();
        int[] hash = new int[10];
        for (int i = 0; i < secretWords.length; i++) {
            char secretWord = secretWords[i];
            char guessWord = guessWords[i];
            if (secretWord == guessWord) {
                bulls++;
            } else {
                if (hash[Character.getNumericValue(secretWord)]-- > 0) {
                    cows++;
                }
                if (hash[Character.getNumericValue(guessWord)]++ < 0) {
                    cows++;
                }
            }
        }
        return bulls + "A" + cows + "B";
    }

}
