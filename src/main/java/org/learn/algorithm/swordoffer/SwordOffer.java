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


        offer.GetUglyNumber_Solution(7);

    }

    /**
     * WC80 二维数组中的查找
     *
     * @param target
     * @param array
     * @return
     */
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
            if (array[left] == array[right]) {
                right--;
            }
            int mid = left + (right - left) / 2;
            if (array[mid] > array[right]) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return array[left];
    }


    /**
     * WC109 栈的压入、弹出序列
     *
     * @param pushA
     * @param popA
     * @return
     */
    public boolean IsPopOrder(int[] pushA, int[] popA) {
        if (pushA == null || popA == null) {
            return false;
        }
        Stack<Integer> stack = new Stack<>();
        int j = 0;
        for (int i = 0; i < pushA.length; i++) {
            stack.push(i);
            while (!stack.isEmpty() && popA[j] == pushA[stack.peek()]) {
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

    /**
     * WC97 二叉树中和为某一值的路径
     *
     * @param root
     * @param target
     * @return
     */
    public ArrayList<ArrayList<Integer>> FindPath(TreeNode root, int target) {
        if (root == null) {
            return new ArrayList<>();
        }
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        intervalFindPath(result, new ArrayList<>(), root, target);
        return result;
    }

    private void intervalFindPath(ArrayList<ArrayList<Integer>> result, ArrayList<Integer> tmp, TreeNode root, int target) {
        tmp.add(root.val);
        if (root.left == null && root.right == null && root.val == target) {
            result.add(new ArrayList<>(tmp));
        } else {
            if (root.left != null) {
                intervalFindPath(result, tmp, root.left, target - root.val);
            }
            if (root.right != null) {
                intervalFindPath(result, tmp, root.right, target - root.val);
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

//        intervalPermutation(result, 0, words);
        return result;
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
            if (num == candidate) {
                count++;
            }
        }
        return 2 * count >= array.length ? candidate : -1;
    }

    public int GetNumberOfK(int[] array, int k) {
        if (array == null || array.length == 0) {
            return 0;
        }
        int left = 0;
        int right = array.length - 1;
        int firstIndex = -1;
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
        firstIndex = left;
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
        if (str == null || str.isEmpty()) {
            return "";
        }
        int len = str.length();
        str += str;
        return str.substring(n, n + len);
    }

    /**
     * WC106 翻转单词序列
     *
     * @param str
     * @return
     */
    public String ReverseSentence(String str) {
        if (str == null || str.isEmpty()) {
            return "";
        }
        String[] words = str.split(" ");
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < words.length; i++) {
            builder.append(new StringBuilder(words[i]).reverse());
            if (i != words.length - 1) {
                builder.append(" ");
            }
        }
        return builder.reverse().toString();
    }


    /**
     * WC76 扑克牌顺子
     *
     * @param numbers
     * @return
     */
    public boolean IsContinuous(int[] numbers) {
        if (numbers == null || numbers.length == 0) {
            return false;
        }
        int max = -1;
        int min = 14;
        int flag = 0;
        for (int number : numbers) {
            if (number == 0) {
                continue;
            }
            if (((flag >> number) & 1) != 0) {
                return false;
            }
            if (max < number) {
                max = number;
            }
            if (number < min) {
                min = number;
            }
            if (max - min >= 5) {
                return false;
            }
            flag |= 1 << number;
        }
        return true;
    }

    /**
     * WC88 孩子们的游戏(圆圈中最后剩下的数)
     *
     * @param n
     * @param m
     * @return
     */
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
        boolean valid = sum > 0 && (sum += Sum_Solution(n - 1)) > 0;
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
        while (num2 != 0) {
            int tmp = num1 ^ num2;
            num2 = (num1 & num2) << 1;
            num1 = tmp;
        }
        return num1;
    }

    public int StrToInt(String str) {
        if (str == null) {
            return 0;
        }
        str = str.trim();
        if (str.isEmpty()) {
            return 0;
        }
        int index = 0;
        int sign = 1;
        char[] words = str.toCharArray();
        if (words[0] == '-' || words[0] == '+') {
            sign = words[0] == '+' ? 1 : -1;
            index = 1;
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
            return new int[]{};
        }
        int[] result = new int[A.length];
        int base = 1;
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
        for (int j = 1; j <= n; j++) {
            dp[0][j] = pattern.charAt(j - 1) == '*' && dp[0][j - 2];
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (str.charAt(i - 1) == pattern.charAt(j - 1) || pattern.charAt(j - 1) == '.') {
                    dp[i][j] = dp[i - 1][j - 1];
                } else if (pattern.charAt(j - 1) == '*') {
                    if (str.charAt(i - 1) != pattern.charAt(j - 2) && pattern.charAt(j - 1) != '.') {
                        dp[i][j] = dp[i][j - 2];
                    } else {
                        dp[i][j] = dp[i - 1][j] || dp[i][j - 1] || dp[i][j - 2];
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
        ListNode slow = pHead;
        ListNode fast = pHead;
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next.next;
            slow = slow.next;
            if (fast == slow) {
                fast = pHead;
                while (fast != slow) {
                    fast = fast.next;
                    slow = slow.next;
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
            ListNode current = pHead.next;
            while (current != null && current.val == pHead.val) {
                current = current.next;
            }
            return deleteDuplication(current);
        }
        pHead.next = deleteDuplication(pHead.next);
        return pHead;
    }

    public TreeLinkNode GetNext(TreeLinkNode pNode) {
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


    public double Power(double base, int exponent) {
        if (exponent == 0) {
            return 1;
        }
        if (exponent < 0) {
            base = 1 / base;
            exponent = -exponent;
        }
        return exponent % 2 == 0 ? Power(base * base, exponent / 2) : base * Power(base * base, exponent / 2);
    }

    public ListNode FindKthToTail(ListNode pHead, int k) {
        if (pHead == null || k <= 0) {
            return null;
        }
        int count = 1;
        ListNode fast = pHead;
        while (fast.next != null) {
            fast = fast.next;
            count++;
        }
        if (count < k) {
            return null;
        }
        ListNode root = new ListNode(0);
        root.next = pHead;
        fast = root;
        for (int i = 0; i < count - k; i++) {
            fast = fast.next;
        }
        return fast.next;
    }

    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param matrix char字符型二维数组
     * @param word   string字符串
     * @return bool布尔型
     */
    public boolean hasPath(char[][] matrix, String word) {
        // write code here
        if (matrix == null || matrix.length == 0) {
            return false;
        }
        int row = matrix.length;

        int column = matrix[0].length;

        boolean[][] used = new boolean[row][column];
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (matrix[i][j] == word.charAt(0) && checkPath(i, j, row, column, matrix, 0, word, used)) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean checkPath(int i, int j, int row, int column, char[][] matrix, int start, String word, boolean[][] used) {
        if (start == word.length()) {
            return true;
        }
        if (i < 0 || i == row || j < 0 || j == column || used[i][j] || matrix[i][j] != word.charAt(start)) {
            return false;
        }
        used[i][j] = true;
        if (checkPath(i - 1, j, row, column, matrix, start + 1, word, used)
                || checkPath(i + 1, j, row, column, matrix, start + 1, word, used)
                || checkPath(i, j - 1, row, column, matrix, start + 1, word, used)
                || checkPath(i, j + 1, row, column, matrix, start + 1, word, used)) {
            return true;
        }
        used[i][j] = false;
        return false;
    }

    public int movingCount(int threshold, int rows, int cols) {
        if (threshold <= 0) {
            return 0;
        }
        boolean[][] used = new boolean[rows][cols];
        return count(used, 0, 0, rows, cols, threshold);
    }

    private int count(boolean[][] used, int i, int j, int rows, int cols, int threshold) {
        if (i < 0 || i == rows || j < 0 || j == cols || used[i][j]) {
            return 0;
        }
        int count = 1;
        used[i][j] = true;
        int val = getNum(i) + getNum(j);
        if (val > threshold) {
            return 0;
        }
        count += count(used, i - 1, j, rows, cols, threshold);
        count += count(used, i + 1, j, rows, cols, threshold);
        count += count(used, i, j - 1, rows, cols, threshold);
        count += count(used, i, j + 1, rows, cols, threshold);
        return count;
    }

    private int getNum(int num) {
        int count = 0;
        while (num != 0) {
            count += num % 10;
            num /= 10;
        }
        return count;
    }


    public int cutRope(int target) {
        if (target == 2) {
            return 1;
        }
        if (target == 3) {
            return 2;
        }
        int[] cut = new int[target + 1];
        cut[0] = 1;
        cut[1] = 1;
        cut[2] = 2;
        cut[3] = 3;
        for (int i = 4; i < cut.length; i++) {
            for (int j = 1; j <= i / 2; j++) {
                cut[i] = Math.max(cut[i], cut[i - j] * cut[j]);
            }
        }
        return cut[target];
    }


    /**
     * WC112 树的子结构
     *
     * @param root1
     * @param root2
     * @return
     */
    public boolean HasSubtree(TreeNode root1, TreeNode root2) {
        if (root1 == null || root2 == null) {
            return false;
        }
        return isSubTree(root1, root2) || HasSubtree(root1.left, root2) || HasSubtree(root1.right, root2);
    }

    private boolean isSubTree(TreeNode root1, TreeNode root2) {
        if (root2 == null) {
            return true;
        }
        if (root1 == null) {
            return false;
        }
        if (root1.val != root2.val) {
            return false;
        }
        return isSubTree(root1.left, root2.left) && isSubTree(root1.right, root2.right);
    }


    /**
     * WC114 和为S的两个数字
     *
     * @param array
     * @param sum
     * @return
     */
    public ArrayList<Integer> FindNumbersWithSum(int[] array, int sum) {
        if (array == null || array.length == 0) {
            return new ArrayList<>();
        }
        ArrayList<Integer> result = new ArrayList<>();
        int left = 0;
        int right = array.length - 1;
        while (left < right) {
            int val = array[left] + array[right];
            if (val < sum) {
                left++;
            } else if (val > sum) {
                right--;
            } else {
                result.add(array[left]);
                result.add(array[right]);
            }
        }
        return result;
    }


    /**
     * WC115 丑数
     *
     * @param index
     * @return
     */
    public int GetUglyNumber_Solution(int index) {
        if (index < 7) {
            return index;
        }
        int idx2 = 0;
        int idx3 = 0;
        int idx5 = 0;
        int[] result = new int[index];
        result[0] = 1;
        int i = 1;
        while (i < index) {
            int val = Math.min(Math.min(result[idx2] * 2, result[idx3] * 3), result[idx5] * 5);
            result[i] = val;
            if (val == result[idx2] * 2) {
                idx2++;
            }
            if (val == result[idx3] * 3) {
                idx3++;
            }
            if (val == result[idx5] * 5) {
                idx5++;
            }
            i++;
        }
        return result[index - 1];

    }


}
