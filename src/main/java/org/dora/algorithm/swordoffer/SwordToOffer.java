package org.dora.algorithm.swordoffer;

import org.dora.algorithm.datastructe.Interval;
import org.dora.algorithm.datastructe.ListNode;
import org.dora.algorithm.datastructe.TreeLinkNode;
import org.dora.algorithm.datastructe.TreeNode;
import sun.plugin2.ipc.IPCFactory;

import java.nio.channels.NonWritableChannelException;
import java.util.*;

/**
 * @author liulu
 * @date 2019/04/24
 */
@Deprecated
public class SwordToOffer {

    /**
     * 字符串中 第一个不重复的字符
     *
     * @param ch
     */
    private int[] num = new int[256];

    public static void main(String[] args) {
        SwordToOffer swordToOffer = new SwordToOffer();
//        swordToOffer.StrToInt("-123");
//        swordToOffer.isNumeric(new char[]{'-', '.', '1', '2', '3'});
        TreeNode root = new TreeNode(0);
        root.left = new TreeNode(1);
        root.right = new TreeNode(2);
//        String serialize = swordToOffer.Serialize(root);
//        System.out.println(serialize);
//        swordToOffer.isValid("()");
        int[] unique = new int[]{2, 2, -1};
//        System.out.println(swordToOffer.permuteUnique(unique).toString());
        char[][] grid = {{'1', '1', '0', '0', '0'}, {'0', '1', '0', '1', '1'}, {'0', '0', '0', '1', '1'}, {'0', '0', '0', '0', '0'}, {'0', '0', '1', '1', '1'}};

//        System.out.println(swordToOffer.gridNum(grid));
        int[] lcs = new int[]{1, 6, 4, 7, 5, 3, 2};
//        swordToOffer.LISiii(lcs);
        int[] minMoney = new int[]{5, 2, 3};
//        swordToOffer.minMoney(minMoney, 20);
//        System.out.println(swordToOffer.restoreIpAddresses("101023"));
//        swordToOffer.trans("h i ", 16);
//        swordToOffer.bigNumSum("1258994789086810959258888307221656691275942378416703768", "7007001981958278066472683068554254815482631701142544497");

//        int result = swordToOffer.minmumNumberOfHost(2, preside);
//        System.out.println(result);
    }

    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param root TreeNode类
     * @return int整型一维数组
     */
    public int[] postorderTraversal(TreeNode root) {
        // write code here
        if (root == null) {
            return null;
        }
        LinkedList<Integer> linkedList = new LinkedList<>();

        Stack<TreeNode> stack = new Stack<>();
        while (!stack.isEmpty() || root != null) {
            if (root != null) {
                linkedList.addFirst(root.val);
                stack.push(root);
                root = root.right;
            } else {
                root = stack.pop();
                root = root.left;
            }
        }
        int[] result = new int[linkedList.size()];
        for (int i = 0; i < linkedList.size(); i++) {
            result[i] = linkedList.get(i);
        }
        return result;
    }

    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param root TreeNode类
     * @return int整型
     */
    public int maxDepth(TreeNode root) {
        // write code here
        if (root == null) {
            return 0;
        }
        return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
    }

    public TreeNode Convert(TreeNode pRootOfTree) {
        if (pRootOfTree == null) {
            return null;
        }
        if (pRootOfTree.left == null && pRootOfTree.right == null) {
            return pRootOfTree;
        }
        TreeNode convert = Convert(pRootOfTree.left);
        if (convert != null) {
            convert.right = pRootOfTree;
        }
        pRootOfTree.left = convert;

        TreeNode convertRight = Convert(pRootOfTree.right);

        pRootOfTree.right = convertRight;
        if (convertRight != null) {
            convertRight.left = pRootOfTree;
        }
        return convert;
    }


    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param root TreeNode类
     * @param p    int整型
     * @param q    int整型
     * @return int整型
     */
    public int lowestCommonAncestor(TreeNode root, int p, int q) {
        // write code here
        if (root == null) {
            return -1;
        }
        if (root.val == p || root.val == q) {
            return root.val;
        }
        if ((p < root.val && root.val < q) || (q < root.val && root.val < p)) {
            return root.val;
        } else if (root.val > p) {
            return lowestCommonAncestor(root.left, p, q);
        } else {
            return lowestCommonAncestor(root.right, p, q);
        }
    }

    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param root TreeNode类
     * @param o1   int整型
     * @param o2   int整型
     * @return int整型
     */
    public int lowestCommonAncestorii(TreeNode root, int o1, int o2) {
        // write code here
        if (root == null) {
            return -1;
        }
        if (root.val == o1 || root.val == o2) {
            return root.val;
        }
        int left = lowestCommonAncestor(root.left, o1, o2);
        int right = lowestCommonAncestor(root.right, o1, o2);
        if (left != -1 && right != -1) {
            return root.val;
        } else if (left == -1) {
            return right;
        } else {
            return left;
        }


    }


    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param t1 TreeNode类
     * @param t2 TreeNode类
     * @return TreeNode类
     */
    public TreeNode mergeTrees(TreeNode t1, TreeNode t2) {
        // write code here
        if (t1 == null && t2 == null) {
            return null;
        }
        if (t1 == null || t2 == null) {
            return t1 == null ? t2 : t1;
        }
        t1.val = t1.val + t2.val;

        t1.left = mergeTrees(t1.left, t2.left);

        t1.right = mergeTrees(t1.right, t2.right);

        return t1;

    }

    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param root TreeNode类
     * @return bool布尔型
     */
    public boolean isValidBST(TreeNode root) {
        // write code here
        if (root == null) {
            return true;
        }
        return isValidBST(Integer.MIN_VALUE, root, Integer.MAX_VALUE);
    }

    private boolean isValidBST(int minValue, TreeNode root, int maxValue) {
        if (root == null) {
            return true;
        }
        if (root.val < minValue || root.val > maxValue) {
            return false;
        }
        return isValidBST(minValue, root.left, root.val) && isValidBST(root.val, root.right, maxValue);
    }


    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param root TreeNode类
     * @return bool布尔型
     */
    public boolean isCompleteTree(TreeNode root) {
        // write code here
        if (root == null) {
            return true;
        }
        if (root.left == null && root.right == null) {
            return true;
        }
        LinkedList<TreeNode> linkedList = new LinkedList<>();
        linkedList.addFirst(root);
        boolean isComplete = false;
        while (!linkedList.isEmpty()) {
            TreeNode poll = linkedList.poll();
            if (poll == null) {
                isComplete = true;
                continue;
            }
            if (isComplete) {
                return false;
            }
            linkedList.add(poll.left);
            linkedList.add(poll.right);
        }
        return true;
    }


    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param pRoot TreeNode类
     * @return bool布尔型
     */
    public boolean IsBalanced_Solution(TreeNode pRoot) {
        // write code here
        if (pRoot == null) {
            return true;
        }
        if (pRoot.left == null && pRoot.right == null) {
            return true;
        }
        int left = maxDepth(pRoot.left);
        int right = maxDepth(pRoot.right);

        if (Math.abs(left - right) > 1) {
            return false;
        }
        return IsBalanced_Solution(pRoot.left) && IsBalanced_Solution(pRoot.right);


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

    private TreeNode buildPreBinaryTree(int preIndex, int[] pre, int inStart, int inEnd, int[] in) {
        if (preIndex >= pre.length || inStart > inEnd) {
            return null;
        }
        TreeNode root = new TreeNode(pre[preIndex]);
        int index = 0;
        for (int i = inStart; i <= inEnd; i++) {
            if (in[i] == root.val) {
                index = i;
                break;
            }
        }
        root.left = buildPreBinaryTree(preIndex + 1, pre, inStart, index - 1, in);
        root.right = buildPreBinaryTree(preIndex + index - inStart + 1, pre, index + 1, inEnd, in);
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
        int tmp = array[i];
        array[i] = array[j];
        array[j] = tmp;
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
            fast = fast.next;
            if (fast == null) {
                return null;
            }
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
     * 数字在排序数组中出现的次数
     *
     * @param array
     * @param k
     * @return
     */
    public int GetNumberOfK(int[] array, int k) {
        if (array == null || array.length == 0) {
            return 0;
        }
        int firstIndex = this.getNumberOfFirst(array, 0, array.length - 1, k);
        int lastIndex = this.getNumberOfLast(array, 0, array.length - 1, k);
        if (firstIndex != -1 && lastIndex != -1) {
            return lastIndex - firstIndex + 1;
        }
        return 0;
    }

    private int getNumberOfLast(int[] array, int start, int end, int target) {
        if (start > end) {
            return -1;
        }
        while (start <= end) {
            int mid = start + (end - start) / 2;

            if (array[mid] < target) {
                start = mid + 1;
            } else if (array[mid] > target) {
                end = mid - 1;
            } else if (mid < end && array[mid + 1] == array[mid]) {
                start = mid + 1;
            } else {
                return mid;
            }
        }
        return -1;
    }

    private int getNumberOfFirst(int[] array, int start, int end, int target) {
        if (start > end) {
            return -1;
        }
        while (start <= end) {
            int mid = start + (end - start) / 2;
            if (array[mid] < target) {
                start = mid + 1;
            } else if (array[mid] > target) {
                end = mid - 1;
            } else if (mid > 0 && target == array[mid - 1]) {
                end = mid - 1;
            } else {
                return mid;
            }
        }
        return -1;
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

//    /**
//     * 二叉搜索树与双向链表
//     *
//     * @param pRootOfTree
//     * @return
//     */
//    public TreeNode Convert(TreeNode pRootOfTree) {
//        if (pRootOfTree == null) {
//            return null;
//        }
//        Stack<TreeNode> stack = new Stack<>();
//        TreeNode prev = null;
//        TreeNode p = null;
//        while (!stack.isEmpty() || pRootOfTree != null) {
//            while (pRootOfTree != null) {
//                stack.push(pRootOfTree);
//                pRootOfTree = pRootOfTree.left;
//            }
//            pRootOfTree = stack.pop();
//            if (prev == null) {
//                prev = pRootOfTree;
//                p = prev;
//            } else {
//                prev.right = pRootOfTree;
//                pRootOfTree.left = prev;
//                prev = pRootOfTree;
//            }
//            pRootOfTree = pRootOfTree.right;
//        }
//        return p;
//    }

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
        char[] chars = str.toCharArray();
        Arrays.sort(chars);
        this.Permutation(ans, 0, chars);
        ans.sort(new Comparator<String>() {
            @Override
            public int compare(String o1, String o2) {
                return o1.compareTo(o2);
            }
        });
        return ans;
    }

    private void Permutation(ArrayList<String> ans, int start, char[] chars) {
        if (start == chars.length - 1) {
            ans.add(String.valueOf(chars));
        }
        for (int i = start; i < chars.length; i++) {
            if (i != start && chars[i] == chars[start]) {
                continue;
            }
            this.swap(chars, i, start);
            this.Permutation(ans, start + 1, chars);
            this.swap(chars, i, start);
        }
    }

    private void swap(char[] array, int i, int k) {
        char tmp = array[i];
        array[i] = array[k];
        array[k] = tmp;
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
     * 分别出现一次的数字
     *
     * @param array
     * @param num1
     * @param num2
     */
    public void FindNumsAppearOnce(int[] array, int[] num1, int[] num2) {
        if (array == null || array.length == 0) {
            return;
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
        int base = 1 << index;
        for (int num : array) {
            if ((base & num) != 0) {
                num1[0] ^= num;
            } else {
                num2[0] ^= num;
            }
        }
    }

    /**
     * 和为S的连续正数序列
     * todo 不解 滑动窗口
     *
     * @param sum
     * @return
     */
    public ArrayList<ArrayList<Integer>> FindContinuousSequence(int sum) {
        if (sum <= 0) {
            return new ArrayList<>();
        }
        ArrayList<ArrayList<Integer>> ans = new ArrayList<>();
        int fast = 2;
        int slow = 1;
        while (slow < fast) {
            int value = (fast + slow) * (fast - slow + 1) / 2;
            if (value == sum) {
                ArrayList<Integer> tmp = new ArrayList<>();
                for (int i = slow; i <= fast; i++) {
                    tmp.add(i);
                }
                ans.add(tmp);
                slow++;
            } else if (value < sum) {
                fast++;
            } else {
                slow++;
            }
        }
        return ans;
    }

    /**
     * 左旋转字符串
     *
     * @param str
     * @param n
     * @return
     */
    public String LeftRotateString(String str, int n) {
        char[] chars = str.toCharArray();

        this.rotate(chars, 0, n - 1);
        this.rotate(chars, n, chars.length - 1);
        this.rotate(chars, 0, chars.length - 1);

        return String.valueOf(chars);
    }

    private void rotate(char[] chars, int start, int end) {
        for (int i = start; i <= (start + end) / 2; i++) {
            this.swap(chars, i, start + end - i);
        }
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
        return max - min <= 4;

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
        int m = str.length;
        int n = pattern.length;
        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[0][0] = true;
        for (int j = 1; j <= n; j++) {
            dp[0][j] = pattern[j - 1] == '*' && dp[0][j - 2];
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (str[i - 1] == pattern[j - 1] || pattern[j - 1] == '.') {
                    dp[i][j] = dp[i - 1][j - 1];
                } else if (pattern[j - 1] == '*') {
                    if (str[i - 1] != pattern[j - 2] && pattern[j - 2] != '.') {
                        dp[i][j] = dp[i][j - 2];
                    } else {
                        dp[i][j] = dp[i - 1][j] || dp[i][j - 2] || dp[i][j - 1];
                    }

                }
            }
        }
        return dp[m][n];
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
     * todo 未解
     *
     * @param num
     * @param size
     * @return
     */
    public ArrayList<Integer> maxInWindows(int[] num, int size) {
        if (num == null || num.length == 0 || size == 0) {
            return new ArrayList<>();
        }
        LinkedList<Integer> linkedList = new LinkedList<>();

        ArrayList<Integer> result = new ArrayList<>();
        for (int i = 0; i < num.length; i++) {
            int index = i - size + 1;

            while (!linkedList.isEmpty() && num[linkedList.peekLast()] < num[i]) {
                linkedList.pollLast();
            }
            linkedList.add(i);

            while (!linkedList.isEmpty() && linkedList.peekFirst() < index) {
                linkedList.poll();
            }
            if (index >= 0) {
                result.add(num[linkedList.peekFirst()]);
            }
        }
        return result;
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
                int index = i * cols + j;
                if (matrix[index] == str[0] && this.verifyPath(matrix, used, i, j, 0, str)) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean verifyPath(char[] matrix, boolean[][] used, int i, int j, int k, char[] str) {
        if (k == str.length) {
            return true;
        }
        if (i < 0 || i >= used.length || j < 0 || j >= used[i].length || used[i][j]) {
            return false;
        }
        int index = i * used[0].length + j;
        if (matrix[index] != str[k]) {
            return false;
        }

        used[i][j] = true;
        boolean veiry = this.verifyPath(matrix, used, i - 1, j, k + 1, str) || this.verifyPath(matrix, used, i + 1, j, k + 1, str) || this.verifyPath(matrix, used, i, j - 1, k + 1, str) || this.verifyPath(matrix, used, i, j + 1, k + 1, str);
        if (veiry) {
            return true;
        }
        used[i][j] = false;
        return false;
    }

    /**
     * 不用加减乘除做加法
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

    /**
     * 数组中重复的数字
     *
     * @param numbers
     * @param length
     * @param duplication
     * @return
     */
    public boolean duplicate(int[] numbers, int length, int[] duplication) {
        if (numbers == null || length == 0) {
            return false;
        }
        return false;
    }

    /**
     * 机器人的运动范围
     *
     * @param threshold
     * @param rows
     * @param cols
     * @return
     */
    public int movingCount(int threshold, int rows, int cols) {
        boolean[][] used = new boolean[rows][cols];

        return this.verify(used, 0, 0, threshold);
    }

    private boolean getMovingValue(int i, int j, int threshold) {
        int sum = 0;
        while (i != 0 || j != 0) {
            sum += i % 10 + j % 10;
            i /= 10;
            j /= 10;
        }
        return sum <= threshold;
    }

    private int verify(boolean[][] used, int i, int j, int threshold) {

        if (i < 0 || i >= used.length || j < 0 || j >= used[i].length) {
            return 0;
        }
        if (used[i][j]) {
            return 0;
        }
        boolean notExceed = this.getMovingValue(i, j, threshold);
        if (!notExceed) {
            return 0;
        }
        used[i][j] = true;
        return this.verify(used, i - 1, j, threshold) +

                this.verify(used, i + 1, j, threshold) +

                this.verify(used, i, j - 1, threshold) +

                this.verify(used, i, j + 1, threshold) + 1;
    }

    public void Insert(char ch) {
        int index = 0;
        for (int i = 0; i < num.length; i++) {
            num[i] = -1;
        }
        if (num[ch - 'a'] == -1) {
            num[ch - 'a'] = index;
        } else {
            num[ch - 'a'] = -2;
        }
        index++;
    }

    //return the first appearence once char in current stringstream
    public char FirstAppearingOnce() {
        int minIndex = Integer.MAX_VALUE;
        char ans = '#';
        for (int i = 0; i < 256; i++) {
            if (num[i] >= 0 && num[i] < minIndex) {
                ans = (char) i;
                minIndex = num[i];
            }
        }
        return ans;
    }


    /**
     * 丑数
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
        int[] ans = new int[index];
        ans[0] = 1;
        int end = 1;
        while (end < index) {
            ans[end] = Math.min(Math.min(ans[idx2] * 2, ans[idx3] * 3), ans[idx5] * 5);
            if (ans[idx2] * 2 == ans[end]) {
                idx2++;
            } else if (ans[idx3] * 3 == ans[end]) {
                idx3++;
            } else if (ans[idx5] * 5 == ans[end]) {
                idx5++;
            }
            end++;
        }
        return ans[index - 1];
    }

    /**
     * 孩子们的游戏(圆圈中最后剩下的数)
     *
     * @param n
     * @param m
     * @return
     */
    public int LastRemaining_Solution(int n, int m) {
        if (n < 1 || m < 1) {
            return -1;
        }
        int[] ans = new int[n];
        int count = n;
        int index = -1;
        int step = 0;
        while (count > 0) {
            index++;
            if (index >= n) {
                index = 0;
            }
            if (ans[index] == -1) {
                continue;
            }
            step++;
            if (step == m) {
                count--;
                ans[index] = -1;
                step = 0;
            }
        }
        return index;
    }

    /**
     * 表示数值的字符串
     *
     * @param str
     * @return
     */
    public boolean isNumeric(char[] str) {
        if (str == null) {
            return false;
        }


        boolean hasE = false;
        boolean hasDecimal = false;
        boolean hasSign = false;
        int index = 0;
        while (index < str.length && !Character.isDigit(str[index])) {
            boolean goodSign = str[index] == '-' || str[index] == '+';
            if (!goodSign) {
                return false;
            }
            index++;
        }
        HashMap<Character, Integer> map = new HashMap<>();
        for (int i = 0; i < str.length; i++) {
            char c = str[i];
            if (c == 'e' || c == 'E') {

                if (i == str.length - 1) {
                    return false;
                }
                if (hasE) {
                    return false;
                }
                hasE = true;
            } else if (c == '-' || c == '+') {

            }
        }
        return true;
    }

    public String Serialize(TreeNode root) {
        StringBuilder stringBuilder = new StringBuilder();
        internalSerialize(root, stringBuilder);
        return stringBuilder.toString();
    }

    private void internalSerialize(TreeNode root, StringBuilder builder) {
        if (root == null) {
            builder.append("#,");
            return;
        }
        builder.append(root.val);
        builder.append(",");
        internalSerialize(root.left, builder);
        internalSerialize(root.right, builder);
    }

    public TreeNode Deserialize(String str) {
        if (str == null || str.isEmpty()) {
            return null;
        }
        LinkedList<String> linkedList = new LinkedList<>();
        String[] items = str.split(",");
        if (items.length == 0) {
            return null;
        }
        Collections.addAll(linkedList, items);
        return internalDeserialize(linkedList);
    }

    private TreeNode internalDeserialize(LinkedList<String> linkedList) {
        if (linkedList.isEmpty()) {
            return null;
        }
        String poll = linkedList.poll();
        if (poll.equals("#")) {
            return null;
        }
        TreeNode node = new TreeNode(Integer.parseInt(poll));
        node.left = internalDeserialize(linkedList);
        node.right = internalDeserialize(linkedList);
        return node;
    }

    private final Stack<Integer> minStack = new Stack<>();
    private final Stack<Integer> normalStack = new Stack<>();

    public void push(int node) {
        normalStack.push(node);
        if (minStack.isEmpty() || node <= minStack.peek()) {
            minStack.push(node);
        }
    }

    public void pop() {
        Integer pop = normalStack.pop();

        if (minStack.isEmpty()) {
            return;
        }
        if (minStack.peek().equals(pop)) {
            minStack.pop();
        }
    }

    public int top() {
        return normalStack.peek();
    }

    public int min() {
        return minStack.peek();
    }


    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param s string字符串
     * @return bool布尔型
     */
    public boolean isValid(String s) {
        // write code here
        if (s == null || s.isEmpty()) {
            return true;
        }
        Stack<Character> stack = new Stack<>();
        char[] words = s.toCharArray();
        for (char word : words) {
            if (word == '[') {
                stack.push(']');
            } else if (word == '{') {
                stack.push('}');
            } else if (word == '(') {
                stack.push(')');
            } else if (stack.isEmpty() || stack.peek() != word) {
                return false;
            } else {
                stack.pop();
            }
        }
        return stack.isEmpty();
    }


    private final PriorityQueue<Integer> big = new PriorityQueue<>();
    private final PriorityQueue<Integer> small = new PriorityQueue<>(Comparator.reverseOrder());

    public void Insert(Integer num) {
        small.offer(num);
        big.offer(small.poll());
        if (big.size() > small.size()) {
            small.offer(big.poll());
        }
    }


    public Double GetMedian() {
        if (small.size() > big.size()) {
            return small.peek() / 1.0;
        }
        return (small.peek() + big.peek()) / 2.0;
    }


    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * <p>
     * 返回表达式的值
     *
     * @param s string字符串 待计算的表达式
     * @return int整型
     */
    public int express(String s) {
        // write code here
        if (s == null || s.isEmpty()) {
            return 0;
        }
        char[] words = s.toCharArray();
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < words.length; i++) {
            char tmp = words[i];

            if (tmp == '(') {
                int end = i;
                int count = 0;
                while (end < words.length) {
                    if (words[end] == '(') {
                        count++;
                    } else if (words[end] == ')') {
                        count--;
                    } else {
                        end++;
                        continue;
                    }
                    if (count == 0) {
                        break;
                    }
                    end++;
                }
                String remain = s.substring(i + 1, end);

                int express = express(remain);
                stack.push(express);
            } else {

            }
        }
        return -1;
    }


    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param numbers int整型一维数组
     * @return int整型
     */
    public int MoreThanHalfNum_Solution(int[] numbers) {
        // write code here
        int base = numbers[0];
        int count = 0;
        for (int number : numbers) {
            if (number == base) {
                count++;
                continue;
            }
            count--;
            if (count == 0) {
                count = 1;
                base = number;
            }
        }
        return base;
//        count = 0;
//        for (int number : numbers) {
//            if (number == base) {
//                count++;
//            }
//        }
//        return

    }


    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param nums int整型一维数组
     * @return int整型一维数组
     */
    public int[] FindNumsAppearOnce(int[] nums) {
        // write code here
        int multi = 0;
        for (int number : nums) {
            multi ^= number;
        }
        multi &= -multi;
        int[] result = new int[2];
        for (int number : nums) {
            if ((number & multi) != 0) {
                result[0] ^= number;
            } else {
                result[1] ^= number;

            }
        }
        return result;
    }

    public int minNumberDisappeared(int[] nums) {
        // write code here
        for (int i = 0; i < nums.length; i++) {
            while (nums[i] > 0 && nums[i] <= nums.length && nums[i] != nums[nums[i] - 1]) {
                swapNum(nums, i, nums[i] - 1);
            }
        }
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != i + 1) {
                return i + 1;

            }
        }
        return num.length + 1;
    }


    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param num int整型一维数组
     * @return int整型ArrayList<ArrayList <>>
     */
    public ArrayList<ArrayList<Integer>> permute(int[] num) {
        // write code here
        if (num == null || num.length == 0) {
            return new ArrayList<>();
        }
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
//        Arrays.sort(num);
        boolean[] used = new boolean[num.length];
        internalPermute(result, new ArrayList<>(), used, num);
        return result;
    }

    private void internalPermute(ArrayList<ArrayList<Integer>> result, ArrayList<Integer> tmp, boolean[] used, int[] num) {
        if (tmp.size() == num.length) {
            result.add(new ArrayList<>(tmp));
            return;
        }
        for (int i = 0; i < num.length; i++) {
            if (used[i]) {
                continue;
            }
            used[i] = true;
            tmp.add(num[i]);
            internalPermute(result, tmp, used, num);
            used[i] = false;
            tmp.remove(tmp.size() - 1);
        }
    }

    /**
     * todo
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param num int整型一维数组
     * @return int整型ArrayList<ArrayList <>>
     */
    public ArrayList<ArrayList<Integer>> permuteUnique(int[] num) {
        // write code here
        if (num == null || num.length == 0) {
            return new ArrayList<>();
        }
        Arrays.sort(num);
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        boolean[] used = new boolean[num.length];
        internalPermuteUnique(result, new ArrayList<>(), used, 0, num);
        return result;
    }

    private void internalPermuteUnique(ArrayList<ArrayList<Integer>> result, ArrayList<Integer> tmp, boolean[] used, int index, int[] num) {
//        if (index >= num.length - 1) {
//            ArrayList<Integer> sub = new ArrayList<>();
//            for (int member : num) {
//                sub.add(member);
//            }
//            result.add(sub);
//            return;
//        }
//        for (int i = index; i < num.length; i++) {
//            if (i > index && num[i] == num[index]) {
//                continue;
//            }
//            if (index > 0 && num[index] == num[index - 1]) {
//                continue;
//            }
//            swapNum(num, i, index);
//            internalPermuteUnique(result, tmp, used, index + 1, num);
//            swapNum(num, i, index);
//        }
        if (tmp.size() == num.length) {
            result.add(new ArrayList<>(tmp));
            return;
        }
        for (int i = 0; i < num.length; i++) {
            if (used[i]) {
                continue;
            }
            if (i > 0 && num[i] == num[i - 1] && !used[i - 1]) {
                continue;
            }
            tmp.add(num[i]);
            used[i] = true;
            internalPermuteUnique(result, tmp, used, index, num);
            used[i] = false;
            tmp.remove(tmp.size() - 1);
        }
    }


    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * <p>
     * 判断岛屿数量
     *
     * @param grid char字符型二维数组
     * @return int整型
     */
    public int gridNum(char[][] grid) {
        // write code here
        if (grid == null || grid.length == 0) {
            return 0;
        }
        int count = 0;
        int row = grid.length;
        int column = grid[0].length;
        boolean[][] used = new boolean[row][column];
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < grid[i].length; j++) {
                if (used[i][j]) {
                    continue;
                }
                if (grid[i][j] == '1') {
                    count++;
                    internalGrid(grid, i, j, used);
                }
            }
        }
        return count;
    }

    private void internalGrid(char[][] grid, int i, int j, boolean[][] used) {
        if (i < 0 || i == grid.length || j < 0 || j == grid[0].length) {
            return;
        }
        if (used[i][j]) {
            return;
        }
        used[i][j] = true;
        if (grid[i][j] == '0') {
            return;
        }
        internalGrid(grid, i - 1, j, used);
        internalGrid(grid, i + 1, j, used);
        internalGrid(grid, i, j - 1, used);
        internalGrid(grid, i, j + 1, used);
    }


    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * <p>
     * 最少货币数
     *
     * @param arr int整型一维数组 the array
     * @param aim int整型 the target
     * @return int整型
     */
    public int minMoney(int[] arr, int aim) {
        // write code here
        if (arr == null || arr.length == 0) {
            return 0;
        }
        int[][] dp = new int[arr.length][aim + 1];
        for (int j = 1; j <= aim; j++) {
            dp[0][j] = Integer.MAX_VALUE;// 无法凑出数值为j的钱
            if (j - arr[0] >= 0 && dp[0][j - arr[0]] != Integer.MAX_VALUE) {
                dp[0][j] = dp[0][j - arr[0]] + 1;// 仅使用第一种类型的货币
            }
        }
        for (int j = 1; j <= aim; j++) {
            for (int i = 1; i < arr.length; i++) {
                if (j - arr[i] >= 0 && dp[i][j - arr[i]] != Integer.MAX_VALUE) {
                    dp[i][j] = Math.min(dp[i - 1][j], 1 + dp[i][j - arr[i]]);
                } else {
                    dp[i][j] = dp[i - 1][j];
                }
            }
        }
        return dp[arr.length - 1][aim] == Integer.MAX_VALUE ? -1 : dp[arr.length - 1][aim];
    }

    public int minMoneyii(int[] arr, int aim) {
        if (arr == null || arr.length == 0) {
            return -1;
        }
        int[] dp = new int[aim + 1];
        Arrays.fill(dp, Integer.MAX_VALUE);
        dp[0] = 0;
        for (int i = 1; i <= aim; i++) {
            int min = Integer.MAX_VALUE;
            for (int money : arr) {
                if (i - money >= 0 && dp[i - money] != Integer.MAX_VALUE) {
                    min = Math.min(min, dp[i - money]);
                }
            }
            dp[i] = min;
        }
        return dp[aim] == Integer.MAX_VALUE ? -1 : dp[aim];
    }


    /**
     * BM71 最长上升子序列(一)
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * <p>
     * 给定数组的最长严格上升子序列的长度。
     *
     * @param arr int整型一维数组 给定的数组
     * @return int整型
     */
    public int LISiii(int[] arr) {
        // write code here
        if (arr == null || arr.length == 0) {
            return 0;
        }
        int[] dp = new int[arr.length];
        Arrays.fill(dp, 1);
        int result = 0;
        for (int i = 1; i < arr.length; i++) {
            for (int j = 0; j < i; j++) {
                if (arr[j] < arr[i] && dp[j] + 1 > dp[i]) {
                    dp[i] = dp[j] + 1;

                    result = Math.max(result, dp[i]);
                }
            }
        }
        return result;
    }


    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param A string字符串
     * @return int整型
     */
    public int getLongestPalindrome(String A) {
        // write code here
        if (A == null || A.isEmpty()) {
            return 0;
        }
        int len = A.length();
        boolean[][] dp = new boolean[len][len];
        int result = 0;
        for (int i = 0; i < len; i++) {
            for (int j = 0; j <= i; j++) {
                if (A.charAt(j) == A.charAt(i) && ((i - j <= 2) || dp[j + 1][i - 1])) {

                    dp[j][i] = true;
                    result = Math.max(result, i - j) + 1;

                }
            }
        }
        return result;
    }

    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param s string字符串
     * @return string字符串ArrayList
     */
    public ArrayList<String> restoreIpAddresses(String s) {
        // write code here
        if (s == null || s.isEmpty()) {
            return new ArrayList<>();
        }
        int len = s.length();
        ArrayList<String> result = new ArrayList<>();
        for (int i = 1; i < 4 && i < len - 2; i++) {
            for (int j = i + 1; j < j + 4 && j < len - 1; j++) {
                for (int k = j + 1; k < k + 4 && k < len; k++) {
                    String a = s.substring(0, i);
                    String b = s.substring(i, j);
                    String c = s.substring(j, k);
                    String d = s.substring(k);
                    if (checkIsValidIp(a) && checkIsValidIp(b) && checkIsValidIp(c) && checkIsValidIp(d)) {
                        result.add(a + "." + b + "." + c + "." + d);
                    }
                }
            }
        }
        return result;
    }

    private boolean checkIsValidIp(String s) {
        if (s == null || s.isEmpty()) {
            return false;
        }
        if (s.length() > 1 && s.charAt(0) == '0') {
            return false;
        }
        int num = Integer.parseInt(s);

        return num >= 0 && num <= 255;
    }

    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param str1 string字符串
     * @param str2 string字符串
     * @return int整型
     */
    public int editDistance(String str1, String str2) {
        // write code here
        int m = str1.length();
        int n = str2.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++) {
            dp[i][0] = i;
        }
        for (int j = 1; j <= n; j++) {
            dp[0][j] = j;
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (str1.charAt(i - 1) == str2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = Math.min(1 + Math.min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]);
                }
            }
        }
        return dp[m][n];
    }

    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param nums int整型一维数组
     * @return int整型
     */
    public int rob(int[] nums) {
        // write code here
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int prev = 0;
        int current = nums[0];
        for (int i = 1; i < nums.length; i++) {
            int tmp = Math.max(prev + nums[i], current);
            prev = current;
            current = tmp;
        }
        return Math.max(current, prev);
    }

    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param nums int整型一维数组
     * @return int整型
     */
    public int robii(int[] nums) {
        // write code here
        if (nums == null || nums.length == 0) {
            return 0;
        }
        return Math.max(internalRob(nums, 0, nums.length - 2), internalRob(nums, 1, nums.length - 1));
    }

    private int internalRob(int[] tmp, int start, int end) {
        if (start > end) {
            return 0;
        }
        int prev = 0;
        int current = tmp[start];
        for (int i = start + 1; i <= end; i++) {
            int val = Math.max(prev + tmp[i], current);
            prev = current;
            current = val;
        }
        return Math.max(current, prev);
    }

    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param s string字符串
     * @param n int整型
     * @return string字符串
     */
    public String trans(String s, int n) {
        // write code here
        int len = s.length();
        char[] words = s.toCharArray();

        int index = 0;
        StringBuilder stringBuilder = new StringBuilder();
        while (index < words.length) {
            char current = words[index];

            if (current == ' ') {
                index++;
                continue;
            }

            if (Character.isLowerCase(current)) {
                words[index] = Character.toUpperCase(current);
//                    tmpBuilder.append(Character.toUpperCase(current));
            } else {
                words[index] = (Character.toLowerCase(current));
            }
            index++;
        }
        s = String.valueOf(words);
        String[] tmp = s.split(" ", -1);
        StringBuilder builder = new StringBuilder();
        for (int i = tmp.length - 1; i >= 0; i--) {
            builder.append(tmp[i]);
            if (i > 0) {
                builder.append(" ");
            }
        }
        return builder.toString();
    }


    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param strs string字符串一维数组
     * @return string字符串
     */
    public String longestCommonPrefix(String[] strs) {
        // write code here
        if (strs == null || strs.length == 0) {
            return "";
        }
        String prefix = strs[0];
        for (int i = 1; i < strs.length; i++) {
            while (strs[i].indexOf(prefix) != 0) {
                prefix = prefix.substring(0, prefix.length() - 1);

            }
        }
        return prefix;
    }


    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * <p>
     * 计算两个数之和
     *
     * @param s string字符串 表示第一个整数
     * @param t string字符串 表示第二个整数
     * @return string字符串
     */
    public String bigNumSum(String s, String t) {
        // write code here
        int m = s.length();
        int n = t.length();
        int carry = 0;
        StringBuilder builder = new StringBuilder();
        while (m >= 0 || n >= 0 || carry > 0) {
            int val1 = (m <= 0 ? 0 : Character.getNumericValue(s.charAt(m - 1)));
            int val2 = (n <= 0 ? 0 : Character.getNumericValue(t.charAt(n - 1)));
            int tmp = carry + val2 + val1;

            int tail = tmp % 10;

            carry = tmp / 10;

            builder.append(tail);

            m--;
            n--;
        }

        String result = builder.reverse().toString();

        if (result.charAt(0) == '0' && result.length() > 1) {
            return result.substring(1);
        }
        return result;
    }


    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param str string字符串 待判断的字符串
     * @return bool布尔型
     */
    public boolean judgePaindrome(String str) {
        // write code here
        if (str == null || str.isEmpty()) {
            return false;
        }
        int start = 0;
        int end = str.length() - 1;
        while (start < end) {
            if (str.charAt(start) != str.charAt(end)) {
                return false;
            }
            start++;
            end--;
        }
        return true;
    }

    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param intervals Interval类ArrayList
     * @return Interval类ArrayList
     */
    public ArrayList<Interval> merge(ArrayList<Interval> intervals) {
        // write code here
        if (intervals == null || intervals.isEmpty()) {
            return new ArrayList<>();
        }
        LinkedList<Interval> result = new LinkedList<>();
        intervals.sort(Comparator.comparingInt(o -> o.start));
        for (Interval interval : intervals) {

            if (result.isEmpty() || result.getLast().end < interval.start) {
                result.add(interval);
            } else {
                Interval last = result.getLast();
                last.end = Math.max(last.end, interval.end);
            }
        }
        return new ArrayList<>(result);
    }

    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param arr int整型一维数组 the array
     * @return int整型
     */
    public int maxLength(int[] arr) {
        // write code here
        if (arr == null || arr.length == 0) {
            return 0;
        }
        int result = 0;
        int left = 0;
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < arr.length; i++) {
            if (map.containsKey(arr[i])) {
                left = Math.max(left, map.get(arr[i]) + 1);
            }
            map.put(arr[i], i);

            result = Math.max(result, i - left + 1);
        }
        return result;
    }


    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * <p>
     * 计算成功举办活动需要多少名主持人
     * BM96 主持人调度（二）
     *
     * @param n        int整型 有n个活动
     * @param startEnd int整型二维数组 startEnd[i][0]用于表示第i个活动的开始时间，startEnd[i][1]表示第i个活动的结束时间
     * @return int整型
     */
    public int minmumNumberOfHost(int n, int[][] startEnd) {
        // write code here

        if (startEnd == null || startEnd.length == 0) {
            return 0;
        }
        Arrays.sort(startEnd, Comparator.comparingInt(o -> o[0]));

        PriorityQueue<Integer> result = new PriorityQueue<>(Integer::compareTo);

        for (int[] current : startEnd) {
            if (!result.isEmpty() && result.peek() <= current[0]) {
                result.poll();
            }
            result.offer(current[1]);
        }
        return result.size();

    }


}