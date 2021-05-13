package org.learn.algorithm.swordoffer;

import org.learn.algorithm.datastructure.ListNode;
import org.learn.algorithm.datastructure.RandomListNode;
import org.learn.algorithm.datastructure.TreeLinkNode;
import org.learn.algorithm.datastructure.TreeNode;

import java.util.*;

/**
 * @author luk
 * @date 2021/5/12
 */
public class SwordOffer {

    public static void main(String[] args) {

        SwordOffer offer = new SwordOffer();

        int[] nums = new int[]{1, 3, 0, 5, 0};

        offer.IsContinuous(nums);
    }

    public boolean Find(int target, int[][] array) {
        if (array == null || array.length == 0) {
            return false;
        }
        int row = array.length;
        int column = array[0].length;
        int i = row - 1;
        int j = 0;
        while (i >= 0 && j < column) {
            int val = array[i][j];
            if (val == target) {
                return true;
            } else if (val < target) {
                j++;
            } else {
                i--;
            }
        }
        return false;
    }

    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param s string字符串
     * @return string字符串
     */
    public String replaceSpace(String s) {
        // write code here
        if (s == null || s.isEmpty()) {
            return "";
        }
        String[] words = s.split(" ", -1);
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < words.length; i++) {
            builder.append(words[i]);
            if (i != words.length - 1) {
                builder.append("%20");
            }
        }
        return builder.toString();
    }

    public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
        if (listNode == null) {
            return new ArrayList<>();
        }
        ArrayList<Integer> integers = printListFromTailToHead(listNode.next);

        integers.add(listNode.val);

        ArrayList<Integer> result = new ArrayList<>(integers);

        return result;
    }

    public TreeNode reConstructBinaryTree(int[] pre, int[] in) {
        if (pre == null || in == null) {
            return null;
        }
        return constructTree(0, pre, 0, in.length - 1, in);
    }

    private TreeNode constructTree(int preStart, int[] pre, int inStart, int inEnd, int[] in) {
        if (preStart == pre.length || inStart > inEnd) {
            return null;
        }
        TreeNode root = new TreeNode(pre[preStart]);
        int index = -1;
        for (int i = inStart; i <= inEnd; i++) {
            if (in[i] == root.val) {
                index = i;
                break;
            }
        }
        root.left = constructTree(preStart + 1, pre, inStart, index - 1, in);
        root.right = constructTree(preStart + index - inStart + 1, pre, index + 1, inEnd, in);
        return root;
    }


    public int minNumberInRotateArray(int[] array) {
        if (array == null || array.length == 0) {
            return 0;
        }
        int left = 0;
        int right = array.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (array[mid] > array[right]) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return array[left];
    }


    public boolean IsPopOrder(int[] pushA, int[] popA) {
        if (pushA == null || popA == null) {
            return false;
        }
        Stack<Integer> stack = new Stack<>();
        int j = 0;
        for (int i = 0; i < pushA.length; i++) {
            stack.push(i);
            while (!stack.isEmpty() && pushA[stack.peek()] == popA[j]) {
                stack.pop();
                j++;
            }
        }
        return stack.isEmpty();
    }

    public ArrayList<Integer> PrintFromTopToBottom(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        LinkedList<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        ArrayList<Integer> result = new ArrayList<>();
        while (!queue.isEmpty()) {

            TreeNode poll = queue.poll();
            result.add(poll.val);
            if (poll.left != null) {
                queue.offer(poll.left);
            }
            if (poll.right != null) {
                queue.offer(poll.right);
            }
        }
        return result;
    }


    public boolean VerifySequenceOfBST(int[] sequence) {
        if (sequence == null || sequence.length == 0) {
            return false;
        }
        return intervalVerify(sequence, 0, sequence.length - 1);
    }

    private boolean intervalVerify(int[] sequence, int start, int end) {
        if (start > end) {
            return false;
        }
        if (start == end) {
            return true;
        }
        int leftIndex = start;
        while (leftIndex < end && sequence[leftIndex] < sequence[end]) {
            leftIndex++;
        }
        int rightIndex = leftIndex;
        while (rightIndex < end && sequence[rightIndex] > sequence[end]) {
            rightIndex++;
        }
        if (rightIndex != end) {
            return false;
        }
        if (leftIndex == start || leftIndex == end) {
            return true;
        }
        return intervalVerify(sequence, start, leftIndex - 1) && intervalVerify(sequence, leftIndex, end - 1);
    }

    public ArrayList<ArrayList<Integer>> FindPath(TreeNode root, int target) {
        if (root == null) {
            return new ArrayList<>();
        }
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        intervalPath(result, new ArrayList<>(), root, target);
        return result;
    }

    private void intervalPath(ArrayList<ArrayList<Integer>> result, ArrayList<Integer> tmp, TreeNode root, int target) {
        tmp.add(root.val);
        if (root.left == null && root.right == null && root.val == target) {
            result.add(new ArrayList<>(tmp));
        } else {
            if (root.left != null) {
                intervalPath(result, tmp, root.left, target - root.val);
            }
            if (root.right != null) {
                intervalPath(result, tmp, root.right, target - root.val);
            }
        }
        tmp.remove(tmp.size() - 1);
    }


    public RandomListNode Clone(RandomListNode pHead) {
        if (pHead == null) {
            return null;
        }
        RandomListNode current = pHead;
        while (current != null) {

            RandomListNode next = current.next;

            RandomListNode node = new RandomListNode(current.label);

            node.next = next;

            current.next = node;

            current = next;
        }
        current = pHead;
        while (current != null) {
            RandomListNode next = current.next;
            if (current.random != null) {
                next.random = current.random.next;
            }
            current = next.next;
        }
        current = pHead;
        RandomListNode copyHead = current.next;
        while (current.next != null) {
            RandomListNode next = current.next;
            current.next = next.next;
            current = next;
        }
        return copyHead;
    }

    public TreeNode Convert(TreeNode pRootOfTree) {
        if (pRootOfTree == null) {
            return null;
        }
        Stack<TreeNode> stack = new Stack<>();
        TreeNode prev = null;
        TreeNode root = null;
        TreeNode p = pRootOfTree;
        while (!stack.isEmpty() || p != null) {
            while (p != null) {
                stack.push(p);
                p = p.left;
            }
            p = stack.pop();
            if (root == null) {
                root = p;
            }
            if (prev != null) {
                prev.right = p;
                p.left = prev;
            }
            prev = p;
            p = p.right;
        }
        return root;
    }


    public ArrayList<String> Permutation(String str) {
        if (str == null || str.isEmpty()) {
            return new ArrayList<>();
        }
        char[] words = str.toCharArray();
        Arrays.sort(words);
        ArrayList<String> result = new ArrayList<>();
        intervalPermutation(result, 0, words);
        result.sort(String::compareTo);
        return result;
    }

    private void intervalPermutation(ArrayList<String> result, int start, char[] words) {
        if (start == words.length) {
            result.add(String.valueOf(words));
            return;
        }
        for (int i = start; i < words.length; i++) {
            if (i > start && words[i] == words[start]) {
                continue;
            }
            swapWord(words, i, start);
            intervalPermutation(result, start + 1, words);
            swapWord(words, i, start);
        }
    }

    private void swapWord(char[] words, int i, int j) {
        char tmp = words[i];
        words[i] = words[j];
        words[j] = tmp;
    }

    public ArrayList<String> PermutationV2(String str) {
        if (str == null) {
            return new ArrayList<>();
        }
        ArrayList<String> result = new ArrayList<>();
        char[] words = str.toCharArray();
        Arrays.sort(words);
        boolean[] used = new boolean[words.length];
        intervalPermutation(result, words, "", used);
        return result;

    }

    private void intervalPermutation(List<String> result, char[] words, String s, boolean[] used) {
        if (s.length() == words.length) {
            result.add(s);
            return;
        }
        for (int i = 0; i < words.length; i++) {
            if (used[i]) {
                continue;
            }
            if (i > 0 && words[i] == words[i - 1] && !used[i - 1]) {
                continue;
            }
            s += words[i];
            used[i] = true;
            intervalPermutation(result, words, s, used);
            s = s.substring(0, s.length() - 1);
            used[i] = false;
        }
    }

    public int MoreThanHalfNum_Solution(int[] array) {
        int candidate = array[0];
        int count = 0;
        for (int num : array) {
            if (num == candidate) {
                count++;
            } else {
                count--;
                if (count == 0) {
                    candidate = num;
                    count = 1;
                }
            }
        }
        count = 0;
        for (int num : array) {
            if (candidate == num) {
                count++;
            }
        }
        return 2 * count > array.length ? candidate : 0;
    }

    public ArrayList<Integer> GetLeastNumbers_Solution(int[] input, int k) {
        if (input == null || input.length == 0 || k <= 0 || k > input.length) {
            return new ArrayList<>();
        }
        PriorityQueue<Integer> priorityQueue = new PriorityQueue<>(Comparator.reverseOrder());
        for (int tmp : input) {
            priorityQueue.offer(tmp);
            if (priorityQueue.size() > k) {
                priorityQueue.poll();
            }
        }
        return new ArrayList<>(priorityQueue);
    }


    public int GetNumberOfK(int[] array, int k) {
        if (array == null || array.length == 0) {
            return 0;
        }
        int left = 0;
        int right = array.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (array[mid] < k) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        if (array[left] != k) {
            return 0;
        }
        int firstIndex = left;
        right = array.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2 + 1;
            if (array[mid] > k) {
                right = mid - 1;
            } else {
                left = mid;
            }
        }
        return left - firstIndex + 1;
    }


    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param array int整型一维数组
     * @return int整型一维数组
     */
    public int[] FindNumsAppearOnce(int[] array) {
        // write code here
        if (array == null || array.length <= 2) {
            return array;
        }
        int result = 0;
        for (int num : array) {
            result ^= num;
        }
        int index = 0;
        for (int i = 0; i < 32; i++) {
            if ((result & (1 << i)) != 0) {
                index = i;
                break;
            }
        }
        int[] ans = new int[2];
        for (int num : array) {
            if ((num & (1 << index)) != 0) {
                ans[0] ^= num;
            } else {
                ans[1] ^= num;
            }
        }
        return ans;
    }


    public String LeftRotateString(String str, int n) {
        int len = str.length();
        if (len < n) {
            return "";
        }
        str += str;
        return str.substring(n, n + len);
    }

    public String ReverseSentence(String str) {
        if (str == null || str.isEmpty()) {
            return "";
        }
        String[] words = str.split(" ");
        StringBuilder builder = new StringBuilder();
        for (int i = words.length - 1; i >= 0; i--) {
            builder.append(words[i]);
            if (i != 0) {
                builder.append(" ");
            }
        }
        return builder.toString();
    }


    public boolean IsContinuous(int[] numbers) {
        if (numbers == null || numbers.length == 0) {
            return false;
        }
        Arrays.sort(numbers);
        int max = 0;
        int min = 14;
        int zeroCount = 0;
        for (int i = 0; i < numbers.length; i++) {
            int val = numbers[i];
            if (val == 0) {
                zeroCount++;
                continue;
            }

            if (i > 0 && val == numbers[i - 1]) {
                return false;
            }
            if (val > max) {
                max = val;
            }
            if (val < min) {
                min = val;
            }
        }
        if (zeroCount >= 4) {
            return true;
        }
        return max - min >= 5;
    }

    public int LastRemaining_Solution(int n, int m) {
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            result.add(i);
        }
        int current = 0;
        while (result.size() > 1) {
            int len = result.size();
            int index = (current + m - 1) % len;
            result.remove(index);
            current = index % (len - 1);
        }
        return result.isEmpty() ? -1 : result.get(0);
    }

    public int Sum_Solution(int n) {
        int sum = n;
        boolean format = sum > 0 && (sum += Sum_Solution(n - 1)) > 0;
        return sum;
    }

    /**
     * todo
     *
     * @param num1
     * @param num2
     * @return
     */
    public int Add(int num1, int num2) {
        return -1;
    }

    public int StrToInt(String str) {
        if (str == null) {
            return 0;
        }
        str = str.trim();
        if (str.isEmpty()) {
            return 0;
        }
        char[] words = str.toCharArray();
        int sign = 1;
        int index = 0;
        if (words[index] == '+' || words[index] == '-') {
            sign = words[index] == '+' ? 1 : -1;
            index++;
        }
        long result = 0;
        while (index < words.length && Character.isDigit(words[index])) {
            result = result * 10 + Character.getNumericValue(words[index]);

            if (result > Integer.MAX_VALUE) {
                return 0;
            }
            index++;
        }
        if (index != words.length) {
            return 0;
        }
        return (int) (result * sign);
    }

    public int[] multiply(int[] A) {
        if (A == null || A.length == 0) {
            return A;
        }
        int len = A.length;
        int base = 1;
        int[] result = new int[len];
        for (int i = 0; i < A.length; i++) {
            result[i] = base;
            base *= A[i];
        }
        base = 1;
        for (int i = A.length - 1; i >= 0; i--) {
            result[i] *= base;
            base *= A[i];
        }
        return result;
    }


    /**
     * todo
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param str     string字符串
     * @param pattern string字符串
     * @return bool布尔型
     */
    public boolean match(String str, String pattern) {
        // write code here
        if (pattern.isEmpty()) {
            return str.isEmpty();
        }
        int m = str.length();
        int n = pattern.length();
        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[0][0] = true;
        for (int i = 1; i <= n; i++) {
            if (pattern.charAt(i - 1) == '*') {
                dp[0][i] = true;
            }
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (str.charAt(i - 1) == pattern.charAt(j - 1) || pattern.charAt(j - 1) == '.') {
                    dp[i][j] = dp[i - 1][j - 1];
                } else if (pattern.charAt(j - 1) == '*') {
                    if (str.charAt(i - 1) == pattern.charAt(j - 2) || pattern.charAt(j - 2) == '.') {
                        dp[i][j] = dp[i][j - 1];
                    } else {
                        dp[i][j] = dp[i - 1][j] || dp[i - 1][j - 1] || dp[i][j - 1];
                    }
                }
            }
        }
        return dp[m][n];
    }

    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param str string字符串
     * @return bool布尔型
     */
    public boolean isNumeric(String str) {
        // write code here
        return false;
    }

    public ListNode EntryNodeOfLoop(ListNode pHead) {
        if (pHead == null || pHead.next == null) {
            return null;
        }
        ListNode fast = pHead;
        ListNode slow = pHead;
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (slow == fast) {
                fast = pHead;
                while (slow != fast) {
                    fast = fast.next;
                    slow = slow.next;
                    ;
                }
                return slow;
            }
        }
        return null;
    }

    public ListNode deleteDuplication(ListNode pHead) {
        if (pHead == null || pHead.next == null) {
            return pHead;
        }
        if (pHead.val == pHead.next.val) {
            ListNode current = pHead.next.next;
            while (current != null && current.val == pHead.val) {
                current = current.next;
            }
            return deleteDuplication(current);
        }
        pHead.next = deleteDuplication(pHead.next);
        return pHead;
    }

    public TreeLinkNode GetNext(TreeLinkNode pNode) {
        if (pNode == null) {
            return null;
        }
        TreeLinkNode right = pNode.right;
        if (right != null) {
            while (right.left != null) {
                right = right.left;
            }
            return right;
        }
        while (pNode.next != null) {
            if (pNode.next.left == pNode) {
                return pNode.next;
            }
            pNode = pNode.next;
        }
        return null;
    }

    public boolean isSymmetrical(TreeNode pRoot) {
        if (pRoot == null) {
            return false;
        }
        return intervalSymmetrical(pRoot.left, pRoot.right);
    }

    private boolean intervalSymmetrical(TreeNode left, TreeNode right) {
        if (left == null && right == null) {
            return true;
        }
        if (left == null || right == null) {
            return false;
        }
        if (left.val != right.val) {
            return false;
        }
        return intervalSymmetrical(left.left, right.right) && intervalSymmetrical(left.right, right.left);
    }


    public int cutRope(int target) {

    }


}
