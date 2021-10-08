package org.learn.algorithm.nowcode;

import org.learn.algorithm.datastructure.ListNode;
import org.learn.algorithm.datastructure.Point;
import org.learn.algorithm.datastructure.TreeLinkNode;
import org.learn.algorithm.datastructure.TreeNode;

import java.util.*;

/**
 * @author luk
 * @date 2021-09-19
 */
public class OftenSolution {

    public static void main(String[] args) {
        OftenSolution solution = new OftenSolution();
        ArrayList<String[]> strings = solution.solveNQueens(4);
        for (String[] word : strings) {
            System.out.println(Arrays.toString(word));
        }
    }


    /**
     * WC1 二叉树的最小深度
     *
     * @param root TreeNode类
     * @return int整型
     */

    public int run(TreeNode root) {
        if (root == null) {
            return 0;
        }
        if (root.left == null && root.right == null) {
            return 1;
        }
        if (root.left == null || root.right == null) {
            return Math.max(run(root.left), run(root.right)) + 1;
        }
        return 1 + Math.min(run(root.left), run(root.right));
    }

    public int runV2(TreeNode root) {
        if (root == null) {
            return 0;
        }
        LinkedList<TreeNode> linkedList = new LinkedList<>();
        linkedList.offer(root);
        int level = 0;
        while (!linkedList.isEmpty()) {
            int size = linkedList.size();
            level++;
            for (int i = 0; i < size; i++) {
                TreeNode poll = linkedList.poll();
                if (poll.left == null && poll.right == null) {
                    return level;
                }
                if (poll.left != null) {
                    linkedList.offer(poll.left);
                }
                if (poll.right != null) {
                    linkedList.offer(poll.right);
                }
            }
        }
        return level;
    }

    /**
     * WC2 后缀表达式求值
     *
     * @param tokens string字符串一维数组
     * @return int整型
     */
    public int evalRPN(String[] tokens) {
        // write code here
        if (tokens == null || tokens.length == 0) {
            return 0;
        }
        Stack<Integer> stack = new Stack<>();
        for (String token : tokens) {
            if ("+".equals(token)) {
                stack.push(stack.pop() + stack.pop());
            } else if ("-".equals(token)) {
                stack.push(-stack.pop() + stack.pop());
            } else if ("*".equals(token)) {
                stack.push(stack.pop() * stack.pop());
            } else if ("/".equals(token)) {
                int divisor = stack.pop();
                int dividend = stack.pop();
                stack.push(dividend / divisor);
            } else {
                stack.push(Integer.parseInt(token));
            }
        }
        return stack.pop();
    }

    /**
     * WC3 多少个点位于同一直线
     *
     * @param points Point类一维数组
     * @return int整型
     */
    public int maxPoints(Point[] points) {
        // write code here
        if (points == null || points.length == 0) {
            return 0;
        }
        int result = 0;
        for (int i = 0; i < points.length; i++) {
            int overload = 0;
            int count = 0;
            Map<Integer, Map<Integer, Integer>> map = new HashMap<>();
            for (int j = i + 1; j < points.length; j++) {
                int x = points[j].x - points[i].x;
                int y = points[j].y - points[i].y;
                if (x == 0 && y == 0) {
                    overload++;
                    continue;
                }
                int gcd = gcd(x, y);
                x /= gcd;
                y /= gcd;
                if (!map.containsKey(x)) {
                    Map<Integer, Integer> tmp = new HashMap<>();
                    tmp.put(y, 1);
                    map.put(x, tmp);
                } else {
                    Map<Integer, Integer> tmp = map.get(x);

                    Integer num = tmp.getOrDefault(y, 0);

                    tmp.put(y, num + 1);
                }
                count = Math.max(count, map.get(x).get(y));
            }
            result = Math.max(result, 1 + overload + count);
        }
        return result;
    }

    private int gcd(int a, int b) {
        if (b == 0) {
            return a;
        }
        return gcd(b, a % b);
    }

    /**
     * todo
     * WC5 链表的插入排序
     *
     * @param head ListNode类
     * @return ListNode类
     */
    public ListNode insertionSortList(ListNode head) {
        // write code here
        if (head == null || head.next == null) {
            return head;
        }
        ListNode dummy = new ListNode(0);
        while (head != null) {
            ListNode pre = dummy;
            while (pre.next != null && pre.next.val < head.val) {
                pre = pre.next;
            }
            ListNode tmp = head.next;
            head.next = pre.next;
            pre.next = head;
            head = tmp;
        }
        return dummy.next;
    }

    /**
     * WC7 求二叉树的前序遍历
     *
     * @param root TreeNode类
     * @return int整型ArrayList
     */
    public ArrayList<Integer> preorderTraversal(TreeNode root) {
        // write code here
        if (root == null) {
            return new ArrayList<>();
        }
        ArrayList<Integer> result = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        stack.add(root);
        while (!stack.isEmpty()) {
            TreeNode pop = stack.pop();
            result.add(pop.val);
            if (pop.right != null) {
                stack.push(pop.right);
            }
            if (pop.left != null) {
                stack.push(pop.left);
            }
        }
        return result;
    }

    /**
     * WC8 重排链表
     *
     * @param head
     */
    public void reorderList(ListNode head) {
        if (head == null || head.next == null) {
            return;
        }
        ListNode slow = head;
        ListNode fast = head;
        while (fast.next != null && fast.next.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode tmp = slow.next;
        slow.next = null;
        ListNode result = head;
        ListNode reverse = reverse(tmp);
        while (result != null && reverse != null) {
            ListNode next = result.next;
            ListNode reverseTmp = reverse.next;
            result.next = reverse;
            reverse.next = next;

            result = next;
            reverse = reverseTmp;
        }
    }

    private ListNode reverse(ListNode head) {
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
     * WC16 词语序列 ii
     * todo
     * 使用图 + bfs
     * http://www.cnblogs.com/ShaneZhang/p/3748494.html
     *
     * @param start
     * @param end
     * @param dict
     * @return
     */
    public ArrayList<ArrayList<String>> findLadders(String start, String end, ArrayList<String> dict) {
        return null;
    }

    /**
     * todo
     * WC17 词语序列
     *
     * @param start
     * @param end
     * @param dict
     * @return
     */
    public int ladderLength(String start, String end, HashSet<String> dict) {
        return 0;
    }

    /**
     * WC19 填充每个节点指向最右节点的next指针 ii
     *
     * @param root
     */
    public void connectII(TreeLinkNode root) {
        if (root == null) {
            return;
        }
        TreeLinkNode current = root;
        while (current != null) {
            TreeLinkNode prev = null;
            TreeLinkNode head = null;
            while (current != null) {
                if (current.left != null) {
                    if (head == null) {
                        head = current.left;
                        prev = head;
                    } else {
                        prev.next = current.left;
                        prev = prev.next;
                    }
                }
                if (current.right != null) {
                    if (head == null) {
                        head = current.right;
                        prev = head;
                    } else {
                        prev.next = current.right;
                        prev = prev.next;
                    }
                }
                current = current.next;
            }
            current = head;
        }
    }

    /**
     * WC51 n-皇后
     *
     * @param n
     * @return
     */
    public ArrayList<String[]> solveNQueens(int n) {
        if (n <= 0) {
            return new ArrayList<>();
        }
        char[][] matrix = new char[n][n];
        for (char[] row : matrix) {
            Arrays.fill(row, '.');
        }
        ArrayList<String[]> result = new ArrayList<>();
        intervalSolveNQueens(result, matrix, 0, n);
        return result;
    }

    private void intervalSolveNQueens(ArrayList<String[]> result, char[][] matrix, int row, int n) {
        if (row == n) {
            String[] tmp = new String[matrix.length];
            for (int i = 0; i < matrix.length; i++) {
                tmp[i] = String.valueOf(matrix[i]);
            }
            result.add(tmp);
            return;
        }
        for (int i = 0; i < n; i++) {
            if (validNQueens(matrix, i, row, n)) {
                matrix[row][i] = 'Q';
                intervalSolveNQueens(result, matrix, row + 1, n);
                matrix[row][i] = '.';
            }
        }
    }

    private boolean validNQueens(char[][] matrix, int column, int row, int n) {
        for (int i = row - 1; i >= 0; i--) {
            if (matrix[i][column] == 'Q') {
                return false;
            }
        }
        for (int i = row - 1, j = column - 1; i >= 0 && j >= 0; i--, j--) {
            if (matrix[i][j] == 'Q') {
                return false;
            }
        }
        for (int i = row - 1, j = column + 1; i >= 0 && j < n; i--, j++) {
            if (matrix[i][j] == 'Q') {
                return false;
            }
        }
        return true;
    }

    /**
     * WC57 搜索插入位置
     *
     * @param A      int整型一维数组
     * @param target int整型
     * @return int整型
     */
    public int searchInsert(int[] A, int target) {
        // write code here
        int left = 0;
        int right = A.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (A[mid] == target) {
                return mid;
            } else if (A[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return left;
    }


    /**
     * WC79 斐波那契数列
     *
     * @param n
     * @return
     */
    public int Fibonacci(int n) {
        if (n <= 1) {
            return n;
        }
        if (n == 2) {
            return 1;
        }
        int sum = 0;
        int tmp1 = 1;
        int tmp2 = 1;
        for (int i = 3; i <= n; i++) {
            sum = tmp1 + tmp2;
            tmp1 = tmp2;
            tmp2 = sum;
        }
        return sum;
    }

    /**
     * WC85 跳台阶
     *
     * @param target
     * @return
     */
    public int jumpFloorII(int target) {
        if (target <= 2) {
            return target;
        }
        return 2 * jumpFloorII(target - 1);
    }


    /**
     * WC108 二叉搜索树的后序遍历序列
     *
     * @param sequence
     * @return
     */
    public boolean VerifySquenceOfBST(int[] sequence) {
        if (sequence == null || sequence.length == 0) {
            return false;
        }
        int endIndex = sequence.length - 1;
        while (endIndex > 0) {
            int index = endIndex - 1;
            while (index >= 0 && sequence[index] > sequence[endIndex]) {
                index--;
            }
            while (index >= 0 && sequence[index] < sequence[endIndex]) {
                index--;
            }
            if (index != -1) {
                return false;
            }
            endIndex--;
        }
        return true;
    }


    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param pRoot TreeNode类
     * @return TreeNode类
     */
    public TreeNode Mirror(TreeNode pRoot) {
        if (pRoot == null) {
            return null;
        }
        TreeNode tmp = pRoot.left;
        pRoot.left = pRoot.right;
        pRoot.right = tmp;
        Mirror(pRoot.left);
        Mirror(pRoot.right);
        return pRoot;
    }

    /**
     * NC41 最长无重复子数组
     *
     * @param arr int整型一维数组 the array
     * @return int整型
     */
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
            map.put(arr[i], i);
            result = Math.max(result, i - left + 1);
        }
        return result;
    }


}
