package org.dora.algorithm.swordoffer;

import org.dora.algorithm.datastructe.ListNode;
import org.dora.algorithm.datastructe.TreeLinkNode;
import org.dora.algorithm.datastructe.TreeNode;

import java.util.*;

/**
 * @author liulu
 * @date 2019/04/24
 */
public class SwordToOffer {
    private void swa

    public static void main(String[] args) {
        SwordToOffer swordToOffer = new SwordToOffer();
        swordToOffer.StrToInt("-123");
    }

    /**
     * 二维数组的查找
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
            if (array[i][j] == target) {
                return true;
            } else if (array[i][j] < target) {
                j++;
            } else {
                i--;
            }
        }
        return false;
    }

    /**
     * 替换空格
     *
     * @param str
     * @return
     */
    public String replaceSpace(StringBuffer str) {
        if (str == null || str.length() == 0) {
            return "";
        }
        String value = str.toString();
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < value.length(); i++) {
            if (value.charAt(i) == ' ') {
                sb.append("%20");
            } else {
                sb.append(value.charAt(i));
            }
        }
        return sb.toString();
    }

    /**
     * 从头到尾打印链表
     *
     * @param listNode
     * @return
     */
    public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
        if (listNode == null) {
            return new ArrayList<>();
        }
        LinkedList<Integer> ans = new LinkedList<>();
        while (listNode != null) {
            ans.addFirst(listNode.val);
            listNode = listNode.next;

        }

        return new ArrayList<>(ans);
    }

    /**
     * 先需
     *
     * @param pre
     * @param in
     * @return
     */
    public TreeNode reConstructBinaryTree(int[] pre, int[] in) {
        if (pre == null || in == null) {
            return null;
        }
        return this.buildPreBinaryTree(0, pre, 0, in.length - 1, in);
    }

    private TreeNode buildPreBinaryTree(int preStart, int[] preorder, int inStart, int inEnd, int[] inOrder) {
        if (preStart >= preorder.length || inStart > inEnd) {
            return null;
        }
        TreeNode root = new TreeNode(preorder[preStart]);
        int index = 0;
        for (int i = inStart; i <= inEnd; i++) {
            if (root.val == inOrder[i]) {
                index = i;
                break;
            }
        }
        root.left = this.buildPreBinaryTree(preStart + 1, preorder, inStart, index - 1, inOrder);
        root.right = this.buildPreBinaryTree(preStart + index - inStart + 1, preorder, index + 1, inEnd, inOrder);
        return root;
    }

    /**
     * 旋转数字的最小数字
     *
     * @param array
     * @return
     */
    public int minNumberInRotateArray(int[] array) {
        if (array == null || array.length == 0) {
            return 0;
        }
        int left = 0;
        int right = array.length - 1;

        // 方案一 和右边界进行比较
//        while (left < right) {
//            int mid = left + (right - left) / 2;
//            if (array[mid] <= array[right]) {
//                right = mid;
//            } else {
//                left = mid + 1;
//            }
//        }

        // 方案二 和左边界进行比较
        while (left < right) {
            if (array[left] < array[right]) {
                return array[left];
            }
            int mid = left + (right - left) / 2;
            if (array[left] <= array[mid]) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return array[left];
    }

    /**
     * 斐波那契数列
     *
     * @param n
     * @return
     */
    public int Fibonacci(int n) {
        if (n == 0) {
            return 0;
        } else if (n <= 2) {
            return 1;
        }
        int sum1 = 1;
        int sum2 = 1;
        int sum3 = 0;
        for (int i = 3; i <= n; i++) {
            sum3 = sum1 + sum2;
            sum1 = sum2;
            sum2 = sum3;
        }
        return sum3;
    }

    /**
     * 跳台阶
     *
     * @param target
     * @return
     */
    public int JumpFloor(int target) {
        if (target == 1) {
            return 1;
        } else if (target == 2) {
            return 2;
        }
        int sum1 = 1;
        int sum2 = 2;
        int sum3 = 0;
        for (int i = 3; i <= target; i++) {
            sum3 = sum1 + sum2;
            sum1 = sum2;
            sum2 = sum3;
        }
        return sum3;
    }

    /**
     * 跳台阶进阶
     *
     * @param target
     * @return
     */
    public int JumpFloorII(int target) {
        if (target == 1) {
            return 1;
        }
        if (target == 2) {
            return 2;
        }
        return 2 * this.JumpFloorII(target - 1);

    }

    /**
     * 数值的整数平方
     *
     * @param base
     * @param exponent
     * @return
     */
    public double Power(double base, int exponent) {
        if (exponent == 0) {
            return 1;
        }
        if (exponent < 0) {
            base = 1 / base;
            exponent = -exponent;
        }
        return (exponent % 2 != 0) ? base * this.Power(base * base, exponent / 2) : this.Power(base * base, exponent / 2);
    }

    /**
     * 调整数组顺序使奇数位于偶数前面
     *
     * @param array
     */
    public void reOrderArray(int[] array) {
        if (array == null || array.length == 0) {
            return;
        }
        for (int i = 0; i < array.length; i++) {
            for (int j = array.length - 1; j > i; j--) {
                if (array[j] % 2 == 1 && array[j - 1] % 2 == 0) {
                    this.swapNum(array, j, j - 1);
                }
            }
        }
    }

    private void swapNum(int[] array, int j, int i) {

    }

    /**
     * 链表中倒数第k个结点
     *
     * @param head
     * @param k
     * @return
     */
    public ListNode FindKthToTail(ListNode head, int k) {
        if (head == null || k <= 0) {
            return null;
        }
        ListNode fast = head;
        for (int i = 0; i < k - 1; i++) {
            if (fast == null) {
                return null;
            }
            fast = fast.next;
        }
        ListNode slow = head;
        while (fast.next != null) {
            fast = fast.next;
            slow = slow.next;
        }
        return slow;
    }

    /**
     * 反转连标
     *
     * @param head
     * @return
     */
    public ListNode ReverseList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        // 非递归
//        ListNode prev = null;
//        ListNode current = head;
//        while (head != null) {
//            ListNode tmp = head.next;
//            head.next = prev;
//            prev = head;
//            head = tmp;
//
//        }
//        return prev;
        ListNode tmp = this.ReverseList(head.next);
        head.next.next = head;
        head.next = null;
        return tmp;

    }


    /**
     * 合并两个排序的链表
     *
     * @param list1
     * @param list2
     * @return
     */
    public ListNode Merge(ListNode list1, ListNode list2) {
        if (list1 == null && list2 == null) {
            return null;
        } else if (list1 == null) {
            return list2;
        } else if (list2 == null) {
            return list1;
        }
        if (list1.val <= list2.val) {
            list1.next = this.Merge(list1.next, list2);
            return list1;
        } else {

            list2.next = this.Merge(list1, list2.next);
            return list2;
        }
    }

    /**
     * 判断是不是子树
     *
     * @param root1
     * @param root2
     * @return
     */
    public boolean HasSubtree(TreeNode root1, TreeNode root2) {
        if (root1 == null || root2 == null) {
            return false;
        }

        return this.isSubTree(root1, root2) || this.HasSubtree(root1.left, root2) || this.HasSubtree(root1.right, root2);
    }

    private boolean isSubTree(TreeNode root, TreeNode node) {
        if (node == null) {
            return true;
        }
        if (root == null) {
            return false;
        }
        if (root.val == node.val) {
            return this.isSubTree(root.left, node.left) && this.isSubTree(root.right, node.right);
        }
        return false;
    }


    /**
     * 二叉树镜像
     *
     * @param root
     */
    public void Mirror(TreeNode root) {
        if (root != null) {
            TreeNode tmp = root.left;
            root.left = root.right;
            root.right = tmp;
            this.Mirror(root.left);
            this.Mirror(root.right);
        }
    }


    /**
     * 矩阵旋转计数
     *
     * @param matrix
     * @return
     */
    public ArrayList<Integer> printMatrix(int[][] matrix) {
        if (matrix == null || matrix.length == 0) {
            return new ArrayList<>();
        }
        ArrayList<Integer> ans = new ArrayList<>();
        int row = matrix.length;
        int column = matrix[0].length;
        int top = 0;
        int left = 0;
        int right = column - 1;
        int bottom = row - 1;
        while (left <= right && top <= bottom) {
            for (int i = left; i <= right; i++) {
                ans.add(matrix[top][i]);
            }
            for (int i = top + 1; i <= bottom; i++) {
                ans.add(matrix[i][right]);
            }
            if (top != bottom) {
                for (int i = right - 1; i >= left; i--) {
                    ans.add(matrix[bottom][i]);
                }
            }
            if (left != right) {
                for (int i = bottom - 1; i > top; i--) {
                    ans.add(matrix[i][left]);
                }
            }
            left++;
            right--;
            top++;
            bottom--;
        }
        return ans;
    }

    /**
     * 栈的压入、弹出序列
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

            stack.push(pushA[i]);

            while (!stack.isEmpty() && stack.peek() == popA[j]) {
                stack.pop();
                j++;
            }
        }
        return stack.isEmpty();
    }

    /**
     * 从上往下打印二叉树
     *
     * @param root
     * @return
     */
    public ArrayList<Integer> PrintFromTopToBottom(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        ArrayList<Integer> ans = new ArrayList<>();
        Deque<TreeNode> deque = new LinkedList<>();
        deque.addLast(root);
        while (!deque.isEmpty()) {
            int size = deque.size();
            for (int i = 0; i < size; i++) {
                TreeNode node = deque.pollFirst();
                if (node.left != null) {
                    deque.addLast(node.left);
                }
                if (node.right != null) {
                    deque.addLast(node.right);
                }
                ans.add(node.val);
            }
        }
        return ans;
    }

    /**
     * 二叉搜索树的后序遍历序列
     *
     * @param sequence
     * @return
     */
    public boolean VerifySquenceOfBST(int[] sequence) {
        if (sequence == null || sequence.length == 0) {
            return false;
        }
        // 方案一 遍历数组 判断元素是否满足BST 顺序
//        int size = sequence.length - 1;
//        int index = 0;
//        while (index < size) {
//            while (index < size && sequence[index] < sequence[size]) {
//                index++;
//            }
//            while (index < size && sequence[index] > sequence[size]) {
//                index++;
//            }
//            if (index != size) {
//                return false;
//            }
//            index = 0;
//            size--;
//        }
        return this.verifyBST(0, sequence.length - 1, sequence);
    }

    private boolean verifyBST(int start, int end, int[] sequence) {
        if (start > end) {
            return false;
        }
        if (start == end) {
            return true;
        }
        int tmp = start;
        while (tmp < end && sequence[tmp] < sequence[end]) {
            tmp++;
        }
        int mid = tmp;

        while (mid < end && sequence[mid] > sequence[end]) {
            mid++;
        }
        if (mid != end) {
            return false;
        }
        if (tmp == start || mid == end) {
            return true;
        }
        return this.verifyBST(start, tmp - 1, sequence) && this.verifyBST(tmp, end - 1, sequence);
    }

    /**
     * 二叉树中和为某一值的路径
     *
     * @param root
     * @param target
     * @return
     */
    public ArrayList<ArrayList<Integer>> FindPath(TreeNode root, int target) {
        if (root == null) {
            return new ArrayList<>();
        }
        ArrayList<ArrayList<Integer>> ans = new ArrayList<>();
        this.findPath(ans, new ArrayList<>(), root, target);
        return ans;
    }

    private void findPath(ArrayList<ArrayList<Integer>> ans, List<Integer> tmp, TreeNode root, int target) {
        tmp.add(root.val);
        if (root.left == null && root.right == null && target == root.val) {
            ans.add(new ArrayList<>(tmp));
        } else {
            if (root.left != null) {
                this.findPath(ans, tmp, root.left, target - root.val);
            }
            if (root.right != null) {
                this.findPath(ans, tmp, root.right, target - root.val);
            }
        }
        tmp.remove(tmp.size() - 1);
    }

    /**
     * 二叉搜索树与双向链表
     *
     * @param pRootOfTree
     * @return
     */
    public TreeNode Convert(TreeNode pRootOfTree) {
        if (pRootOfTree == null) {
            return null;
        }
        Stack<TreeNode> stack = new Stack<>();
        TreeNode prev = null;
        TreeNode p = null;
        while (!stack.isEmpty() || pRootOfTree != null) {
            while (pRootOfTree != null) {
                stack.push(pRootOfTree);
                pRootOfTree = pRootOfTree.left;
            }
            pRootOfTree = stack.pop();
            if (prev == null) {
                prev = pRootOfTree;
                p = prev;
            } else {
                prev.right = pRootOfTree;
                pRootOfTree.left = prev;
                prev = pRootOfTree;
            }
            pRootOfTree = pRootOfTree.right;
        }
        return p;
    }

    /**
     * 字符串排列
     *
     * @param str
     * @return
     */
    public ArrayList<String> Permutation(String str) {
        if (str == null || str.isEmpty()) {
            return new ArrayList<>();
        }
        ArrayList<String> ans = new ArrayList<>();
        this.Permutation(ans, 0, str.toCharArray());
        return ans;
    }

    private void Permutation(ArrayList<String> ans, int start, char[] array) {
        if (start == array.length - 1) {
            ans.add(String.valueOf(array));
        }
        for (int i = start; i < array.length; i++) {
            if (i != start && array[i] == array[start]) {
                continue;
            }
            this.swap(array, i, start);
            this.Permutation(ans, start + 1, array);
            this.swap(array, i, start);
        }
    }

    private void swap(char[] array, int i, int k) {
        char tmp = array[i];
        array[i] = array[k];
        array[k] = tmp;
    }


    /**
     * 数组中出现次数超过一半的数字
     *
     * @param array
     * @return
     */
    public int MoreThanHalfNum_Solution(int[] array) {
        if (array == null || array.length == 0) {
            return -1;
        }
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int num : array) {
            int count = map.getOrDefault(num, 0);
            map.put(num, ++count);
        }
        for (int num : array) {
            int size = map.get(num);
            if (size * 2 > array.length) {
                return num;
            }
        }
        return -1;
    }

    /**
     * 最小的K个数
     *
     * @param input
     * @param k
     * @return
     */
    public ArrayList<Integer> GetLeastNumbers_Solution(int[] input, int k) {
        if (input == null || input.length == 0) {
            return new ArrayList<>();
        }
        if (k <= 0 || k > input.length) {
            return new ArrayList<>();
        }
        k--;
        int start = 0;
        int end = input.length - 1;

        int partition = this.partition(input, start, end);
        while (partition != k) {
            if (partition > k) {
                partition = this.partition(input, start, partition - 1);
            } else {
                partition = this.partition(input, partition + 1, end);
            }
        }
        ArrayList<Integer> ans = new ArrayList<>();
        for (int i = 0; i <= k; i++) {
            ans.add(input[i]);
        }
        return ans;
    }

    public int partition(int[] array, int start, int end) {
        int pivot = array[start];
        while (start < end) {
            while (start < end && array[end] >= pivot) {
                end--;
            }
            if (start < end) {
                array[start] = array[end];
                start++;
            }
            while (start < end && array[start] < pivot) {
                start++;
            }
            if (start < end) {
                array[end] = array[start];
                end--;
            }
        }
        array[start] = pivot;
        return start;
    }

    /**
     * 连续子数组的最大和
     *
     * @param array
     * @return
     */
    public int FindGreatestSumOfSubArray(int[] array) {
        if (array == null || array.length == 0) {
            return -1;
        }
        int local = array[0];
        int global = array[0];
        for (int i = 1; i < array.length; i++) {
            local = local > 0 ? local + array[i] : array[i];
            global = Math.max(global, local);
        }
        return global;
    }

    /**
     * 整数中1出现的次数（从1到n整数中1出现的次数）
     * todo 不懂
     *
     * @param n
     * @return
     */
    public int NumberOf1Between1AndN_Solution(int n) {
        return -1;
    }

    /**
     * 把数组排成最小的数
     *
     * @param numbers
     * @return
     */
    public String PrintMinNumber(int[] numbers) {
        if (numbers == null || numbers.length == 0) {
            return "";
        }
        List<Integer> ans = new ArrayList<>();
        for (int num : numbers) {
            ans.add(num);
        }
        ans.sort(new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                String value1 = o1 + "" + o2;
                String value2 = o2 + "" + o1;
                return value1.compareTo(value2);
            }
        });
        String str = "";
        for (Integer num : ans) {
            str += num;
        }
        return str;

    }

    /**
     * 丑数
     * todo 不懂
     *
     * @param index
     * @return
     */
    public int GetUglyNumber_Solution(int index) {
        return -1;
    }

    /**
     * 第一个只出现一次的字符
     * todo 不懂
     *
     * @param str
     * @return
     */
    public int FirstNotRepeatingChar(String str) {
        int[] num = new int[256];
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < str.length(); i++) {
            num[str.charAt(i) - 'a']++;

        }
        return -1;
    }

    /**
     * 数组中的逆序对
     * todo 不懂
     *
     * @param array
     * @return
     */
    public int InversePairs(int[] array) {
        return -1;
    }

    /**
     * 两个链表的第一个公共结点
     *
     * @param pHead1
     * @param pHead2
     * @return
     */
    public ListNode FindFirstCommonNode(ListNode pHead1, ListNode pHead2) {
        if (pHead1 == null || pHead2 == null) {
            return null;
        }
        ListNode p1 = pHead1;
        ListNode p2 = pHead2;
        while (p1 != p2) {
            p1 = p1 == null ? pHead2 : p1.next;
            p2 = p2 == null ? pHead1 : p2.next;
        }
        return p1;
    }

    /**
     * 二叉树深度
     *
     * @param root
     * @return
     */
    public int TreeDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        return 1 + Math.max(this.TreeDepth(root.left), this.TreeDepth(root.right));
    }

    /**
     * 判断是否是平衡二叉树
     *
     * @param root
     * @return
     */
    public boolean IsBalanced_Solution(TreeNode root) {
        if (root == null) {
            return true;
        }
        int leftDepth = this.TreeDepth(root.left);
        int rightDepth = this.TreeDepth(root.right);
        if (Math.abs(leftDepth - rightDepth) <= 1) {
            return this.IsBalanced_Solution(root.left) && this.IsBalanced_Solution(root.right);
        }
        return false;
    }

    /**
     * 分别出现一次的数字
     *
     * @param array
     * @param num1
     * @param num2
     */
    public void FindNumsAppearOnce(int[] array, int num1[], int num2[]) {

    }


    /**
     * 和为S的连续正数序列
     * todo 不解 滑动窗口
     *
     * @param sum
     * @return
     */
    public ArrayList<ArrayList<Integer>> FindContinuousSequence(int sum) {
        return null;
    }

    /**
     * 左旋转字符串
     *
     * @param str
     * @param n
     * @return
     */
    public String LeftRotateString(String str, int n) {
        str = str + str;
        str = str.substring(n);
        return str;
    }

    /**
     * 翻转单词顺序列
     *
     * @param str
     * @return
     */
    public String ReverseSentence(String str) {
        if (str == null) {
            return "";
        }

        StringBuilder sb = new StringBuilder();
        String[] strs = str.split(" ");
        for (int i = strs.length - 1; i >= 0; i--) {
            sb.append(strs[i]);
            if (i > 0) {
                sb.append(" ");
            }
        }
        return sb.length() != 0 ? sb.toString() : str;
    }

    /**
     * 扑克牌顺子
     *
     * @param numbers
     * @return
     */
    public boolean isContinuous(int[] numbers) {
        if (numbers == null || numbers.length == 0) {
            return false;
        }
        int min = -1;
        int max = 14;
        int countOfZero = 0;
        int[] hash = new int[15];
        for (int num : numbers) {
            if (num == 0) {
                countOfZero++;
                continue;
            }
            hash[num]++;
            if (hash[num] > 1) {
                return false;
            }
            if (num < min) {
                min = num;
            } else if (num > max) {
                max = num;
            }
        }
        if (countOfZero >= 5) {
            return true;
        }
        if (max - min > 4) {
            return false;
        }
        return true;

    }


    /**
     * 求1+2+3+...+n
     *
     * @param n
     * @return
     */
    public int Sum_Solution(int n) {
        n = n > 0 ? n + this.Sum_Solution(n - 1) : 0;
        return n;
    }


    /**
     * 把字符串转换成整数
     *
     * @param str
     * @return
     */
    public int StrToInt(String str) {
        if (str == null) {
            return 0;
        }
        str = str.trim();
        if (str.isEmpty()) {
            return 0;
        }
        int sign = 1;

        int index = 0;
        while (index < str.length() && !Character.isDigit(str.charAt(index))) {
            if (str.charAt(index) == '+') {
                sign = 1;
            }
            if (str.charAt(index) == '-') {
                sign = -1;
            }
            index++;
        }
        long ans = 0;
        while (index < str.length() && Character.isDigit(str.charAt(index))) {
            int value = str.charAt(index) - '0';
            ans = ans * 10 + value;
            index++;
        }
        if (index < str.length()) {
            return 0;
        }
        ans = ans * sign;
        if (ans > Integer.MAX_VALUE) {
            return 0;
        }
        return (int) ans * sign;
    }

    /**
     * 构建乘积数组
     *
     * @param A
     * @return
     */
    public int[] multiply(int[] A) {
        if (A == null || A.length == 0) {
            return new int[]{};
        }
        int[] ans = new int[A.length];
        int base = 1;
        for (int i = 0; i < ans.length; i++) {
            ans[i] = base;
            base *= A[i];
        }
        base = 1;
        for (int i = ans.length - 1; i >= 0; i--) {
            ans[i] *= i;
            base *= A[i];
        }
        return ans;
    }


    /**
     * 正则表达式匹配
     *
     * @param str
     * @param pattern
     * @return
     */
    public boolean match(char[] str, char[] pattern) {
        if (str == null || pattern == null) {
            return false;
        }
        if (pattern.length == 0) {
            return true;
        }
        return false;
    }

    /**
     * 环的指定入口
     *
     * @param pHead
     * @return
     */
    public ListNode EntryNodeOfLoop(ListNode pHead) {
        if (pHead == null || pHead.next == null) {
            return null;
        }
        ListNode fast = pHead;
        ListNode slow = pHead;
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (fast == slow) {
                fast = pHead;
                while (fast != slow) {
                    fast = fast.next;
                    slow = slow.next;
                }
                return fast;
            }
        }
        return null;
    }


    /**
     * 删除重复结点
     *
     * @param pHead
     * @return
     */
    public ListNode deleteDuplication(ListNode pHead) {
        if (pHead == null || pHead.next == null) {
            return pHead;
        }
        if (pHead.val == pHead.next.val) {
            ListNode node = pHead.next.next;
            while (node != null && node.val == pHead.val) {
                node = node.next;
            }
            return this.deleteDuplication(node);
        } else {
            pHead.next = this.deleteDuplication(pHead.next);
            return pHead;
        }

    }


    /**
     * 二叉树的下一个结点
     *
     * @param pNode
     * @return
     */
    public TreeLinkNode GetNext(TreeLinkNode pNode) {
        if (pNode == null) {
            return pNode;
        }
        if (pNode.right != null) {
            TreeLinkNode node = pNode.right;
            while (node.left != null) {
                node = node.left;
            }
            return node;
        }
        while (pNode.next != null) {
            if (pNode.next.left == pNode) {
                return pNode.next;
            }
            pNode = pNode.next;
        }
        return null;
    }

    /**
     * 对称二叉树
     *
     * @param pRoot
     * @return
     */
    boolean isSymmetrical(TreeNode pRoot) {
        if (pRoot == null) {
            return true;
        }
        if (pRoot.left == null && pRoot.right == null) {
            return true;
        }

        if (pRoot.left == null || pRoot.right == null) {
            return false;
        }
        if (pRoot.left.val == pRoot.right.val) {
            return this.isSymme(pRoot.left, pRoot.right);
        }
        return false;
    }

    private boolean isSymme(TreeNode left, TreeNode right) {
        if (left == null && right == null) {
            return true;
        }
        if (left == null || right == null) {
            return false;
        }
        if (left.val == right.val) {
            return this.isSymme(left.left, right.right) && this.isSymme(left.right, right.left);
        }
        return false;
    }


    /**
     * 之字形打印二叉树
     *
     * @param pRoot
     * @return
     */
    public ArrayList<ArrayList<Integer>> Print(TreeNode pRoot) {
        if (pRoot == null) {
            return new ArrayList<>();
        }
        ArrayList<ArrayList<Integer>> ans = new ArrayList<>();
        Deque<TreeNode> deque = new LinkedList<>();
        deque.add(pRoot);
        boolean leftToRight = true;
        while (!deque.isEmpty()) {
            int size = deque.size();
            LinkedList<Integer> tmp = new LinkedList<>();
            for (int i = 0; i < size; i++) {
                TreeNode node = deque.poll();
                if (leftToRight) {
                    tmp.addLast(node.val);
                } else {
                    tmp.addFirst(node.val);
                }
                if (node.left != null) {
                    deque.add(node.left);
                }
                if (node.right != null) {
                    deque.add(node.right);
                }
            }
            ans.add(new ArrayList<>(tmp));
            leftToRight = !leftToRight;
        }
        return ans;
    }


    /**
     * 把二叉树打印成多行
     *
     * @param pRoot
     * @return
     */
    ArrayList<ArrayList<Integer>> PrintII(TreeNode pRoot) {
        if (pRoot == null) {
            return new ArrayList<>();
        }
        ArrayList<ArrayList<Integer>> ans = new ArrayList<>();
        Deque<TreeNode> deque = new LinkedList<>();
        deque.add(pRoot);
        while (!deque.isEmpty()) {
            int size = deque.size();
            LinkedList<Integer> tmp = new LinkedList<>();
            for (int i = 0; i < size; i++) {
                TreeNode node = deque.poll();
                tmp.addLast(node.val);
                if (node.left != null) {
                    deque.add(node.left);
                }
                if (node.right != null) {
                    deque.add(node.right);
                }
            }
            ans.add(new ArrayList<>(tmp));
        }
        return ans;
    }

    /**
     * 二叉搜索树的第k个结点
     *
     * @param pRoot
     * @param k
     * @return
     */
    TreeNode KthNode(TreeNode pRoot, int k) {
        if (pRoot == null || k <= 0) {
            return null;
        }
        Stack<TreeNode> stack = new Stack<>();
        int size = 0;
        while (!stack.isEmpty() || pRoot != null) {
            while (pRoot != null) {
                stack.push(pRoot);
                pRoot = pRoot.left;
            }
            size++;
            pRoot = stack.pop();
            if (size == k) {
                return pRoot;
            }
            pRoot = pRoot.right;
        }
        return null;
    }

    /**
     * 滑动窗口的最大值
     *
     * @param num
     * @param size
     * @return
     */
    public ArrayList<Integer> maxInWindows(int[] num, int size) {
        if (num == null || num.length == 0 || num.length < size) {
            return new ArrayList<>();
        }
        int global = Integer.MIN_VALUE;
        ArrayList<Integer> ans = new ArrayList<>();
        PriorityQueue<Integer> priorityQueue = new PriorityQueue<>(size);
        int begin = 0;
        int end = 0;
        while (end < num.length) {
            int local = Integer.MIN_VALUE;
            while (end - begin <= size - 1) {

                priorityQueue.add(num[end++]);

            }
            int max = priorityQueue.peek();

            begin++;
            ans.add(local);
        }
        return ans;

    }

    /**
     * 矩阵中的路径
     *
     * @param matrix
     * @param rows
     * @param cols
     * @param str
     * @return
     */
    public boolean hasPath(char[] matrix, int rows, int cols, char[] str) {
        if (matrix == null || matrix.length == 0) {
            return false;
        }
        boolean[][] used = new boolean[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int index = i * rows + j * cols;
                if (matrix[i * +j] == str[0] && verify(matrix, i, j, str, 0)) {
                    return true;
                }
            }
        }
        return false;
    }


}
