package org.dora.algorithm.geeksforgeek;

import org.dora.algorithm.datastructe.Node;
import org.dora.algorithm.datastructe.TreeNode;

import java.util.*;

/**
 * 各种遍历方法
 *
 * @author dora
 * @date 2019/11/5
 */
public class Traversal {

    /**
     * 79. Word Search
     * <p>
     * 深度优先遍历
     *
     * @param board
     * @param word
     * @return
     */
    public boolean exist(char[][] board, String word) {
        if (board == null || board.length == 0) {
            return false;
        }
        int row = board.length;
        int column = board[0].length;
        boolean[][] used = new boolean[row][column];
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (board[i][j] == word.charAt(0) && this.checkExist(used, board, i, j, 0, word)) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean checkExist(boolean[][] used, char[][] board,
                               int i, int j, int index, String word) {
        if (index == word.length()) {
            return true;
        }
        if (i < 0 || i >= board.length || j < 0
                || j >= board[0].length || used[i][j] || board[i][j] != word.charAt(index)) {
            return false;
        }
        used[i][j] = true;
        if (this.checkExist(used, board, i - 1, j, index + 1, word) ||
                this.checkExist(used, board, i + 1, j, index + 1, word) ||
                this.checkExist(used, board, i, j - 1, index + 1, word) ||
                this.checkExist(used, board, i, j + 1, index + 1, word)) {
            return true;
        }
        used[i][j] = false;
        return false;
    }


    /**
     * 94. Binary Tree Inorder Traversal
     *
     * @param root
     * @return
     */
    public List<Integer> inorderTraversal(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        List<Integer> ans = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        TreeNode p = root;
        while (!stack.isEmpty() || p != null) {
            while (p != null) {
                stack.push(p);
                p = p.left;
            }
            p = stack.pop();
            ans.add(p.val);
            p = p.right;
        }
        return ans;
    }


    /**
     * 98. Validate Binary Search Tree
     *
     * @param root
     * @return
     */
    public boolean isValidBST(TreeNode root) {
        if (root == null) {
            return true;
        }
        Stack<TreeNode> stack = new Stack<>();
        TreeNode p = root;
        TreeNode prev = null;
        while (!stack.isEmpty() || p != null) {
            while (p != null) {
                stack.push(p);
                p = p.left;
            }
            p = stack.pop();

            if (prev != null && prev.val >= p.val) {
                return false;
            }
            prev = p;

            p = p.right;
        }
        return true;
    }

    /**
     * 99. Recover Binary Search Tree
     *
     * @param root
     */
    public void recoverTree(TreeNode root) {
        if (root == null) {
            return;
        }
        Stack<TreeNode> stack = new Stack<>();
        TreeNode p = root;
        TreeNode prev = null;
        TreeNode first = null;
        TreeNode second = null;
        while (!stack.isEmpty() || p != null) {
            while (p != null) {
                stack.push(p);
                p = p.left;
            }
            p = stack.pop();

            if (prev != null) {
                if (first == null && prev.val >= p.val) {
                    first = prev;
                }
                if (first != null && prev.val >= p.val) {
                    second = p;
                }
            }
            prev = p;
            p = p.right;
        }
        if (first != null) {
            int val = first.val;
            first.val = second.val;
            second.val = val;
        }
    }


    /**
     * 102. Binary Tree Level Order Traversal
     */
    public List<List<Integer>> levelOrder(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        List<List<Integer>> ans = new ArrayList<>();

        Deque<TreeNode> deque = new LinkedList<>();

        deque.push(root);

        while (!deque.isEmpty()) {

            int size = deque.size();

            List<Integer> tmp = new ArrayList<>();

            for (int i = 0; i < size; i++) {

                TreeNode poll = deque.poll();

                tmp.add(poll.val);

                if (poll.left != null) {
                    deque.add(poll.left);
                }

                if (poll.right != null) {
                    deque.add(poll.right);
                }
            }
            ans.add(tmp);
        }
        return ans;
    }

    /**
     * 103. Binary Tree Zigzag Level Order Traversal
     *
     * @param root
     * @return
     */
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        List<List<Integer>> ans = new ArrayList<>();

        Deque<TreeNode> list = new LinkedList<>();

        list.add(root);

        boolean leftToRight = true;

        while (!list.isEmpty()) {
            LinkedList<Integer> tmp = new LinkedList<>();
            int size = list.size();
            for (int i = 0; i < size; i++) {
                TreeNode poll = list.poll();
                if (leftToRight) {
                    tmp.addLast(poll.val);
                } else {
                    tmp.addFirst(poll.val);
                }
                if (poll.left != null) {
                    list.add(poll.left);
                }
                if (poll.right != null) {
                    list.add(poll.right);
                }

            }
            ans.add(tmp);
            leftToRight = !leftToRight;
        }
        return ans;
    }

    /**
     * 107. Binary Tree Level Order Traversal II
     *
     * @param root
     * @return
     */
    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        LinkedList<List<Integer>> ans = new LinkedList<>();

        Deque<TreeNode> deque = new LinkedList<>();

        deque.add(root);

        while (!deque.isEmpty()) {
            List<Integer> tmp = new ArrayList<>();

            int size = deque.size();

            for (int i = 0; i < size; i++) {
                TreeNode poll = deque.poll();
                tmp.add(poll.val);
                if (poll.left != null) {
                    deque.add(poll.left);
                }
                if (poll.right != null) {
                    deque.add(poll.right);
                }
            }
            ans.addFirst(tmp);
        }
        return ans;
    }


    /**
     * 114. Flatten Binary Tree to Linked List
     *
     * @param root
     */
    public void flatten(TreeNode root) {
        if (root == null) {
            return;
        }
        Stack<TreeNode> stack = new Stack<>();


        TreeNode p = root;

        stack.push(p);

        TreeNode prev = null;

        while (!stack.isEmpty()) {
            p = stack.pop();
            if (p.right != null) {
                stack.push(p.right);
            }
            if (p.left != null) {
                stack.push(p.left);
            }

            if (prev != null) {
                prev.right = p;

                prev.left = null;
            }
            prev = p;
        }
    }


    /**
     * todo
     * 115. Distinct Subsequences
     *
     * @param s
     * @param t
     * @return
     */
    public int numDistinct(String s, String t) {
        if (s == null || t == null) {
            return 0;
        }
        int m = s.length();

        int n = t.length();

        int[][] dp = new int[m + 1][n + 1];

        for (int i = 1; i <= m; i++) {
            dp[i][0] = 1;
        }

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                dp[i][j] = (s.charAt(i - 1) == t.charAt(j - 1) ? dp[i - 1][j - 1] : 0) + dp[i - 1][j];
            }
        }
        return dp[m][n];
    }


    /**
     * 116. Populating Next Right Pointers in Each Node
     *
     * @param root
     * @return
     */
    public Node connect(Node root) {
        if (root == null || root.left == null) {
            return root;
        }
        Node p = root;

        while (p.left != null) {

            Node nextNode = p.left;

            while (p != null) {
                p.left.next = p.right;

                if (p.next != null) {

                    p.right.next = p.next.left;
                }
                p = p.next;
            }
            p = nextNode;
        }
        return root;
    }


    /**
     * todo O1空间
     * 关键点在于: 找到并设置每一层的头结点
     * 117. Populating Next Right Pointers in Each Node II
     *
     * @param root
     * @return
     */
    public Node connectII(Node root) {
        if (root == null) {
            return null;
        }
        Node p = root;

        Node levelHead = null;

        Node levelPrev = null;

        while (p != null) {

            while (p != null) {
                if (p.left != null) {
                    if (levelPrev != null) {
                        levelPrev.next = p.left;
                    } else {
                        levelHead = p.left;
                    }
                    levelPrev = p.left;
                }
                if (p.right != null) {
                    if (levelPrev != null) {
                        levelPrev.next = p.right;
                    } else {
                        levelHead = p.right;
                    }
                    levelPrev = p.right;
                }
                p = p.next;
            }
            p = levelHead;

            levelHead = null;

            levelPrev = null;

        }
        return root;
    }

    /**
     * 118. Pascal's Triangle
     *
     * @param numRows
     * @return
     */
    public List<List<Integer>> generate(int numRows) {
        if (numRows < 0) {
            return new ArrayList<>();
        }
        List<List<Integer>> ans = new ArrayList<>();

        for (int i = 0; i <= numRows - 1; i++) {

            List<Integer> tmp = new ArrayList<>();

            tmp.add(1);


            for (int j = 1; j < i; j++) {
                int value = ans.get(i - 1).get(j - 1) + ans.get(i - 1).get(j);

                tmp.add(value);

            }

            if (i > 0) {
                tmp.add(1);
            }
            ans.add(tmp);
        }
        return ans;
    }


    /**
     * 119. Pascal's Triangle II
     *
     * @param rowIndex
     * @return
     */
    public List<Integer> getRow(int rowIndex) {
        if (rowIndex < 0) {
            return new ArrayList<>();
        }
        List<Integer> ans = new ArrayList<>();
        ans.add(1);
        for (int i = 0; i <= rowIndex; i++) {

            for (int j = i - 1; j >= 1; j--) {
                int value = ans.get(j) + ans.get(j - 1);
                ans.set(j, value);
            }
            if (i > 0) {
                ans.add(1);
            }
        }
        return ans;
    }


}
