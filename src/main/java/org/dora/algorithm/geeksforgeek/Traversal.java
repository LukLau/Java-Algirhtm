package org.dora.algorithm.geeksforgeek;

import org.dora.algorithm.datastructe.ListNode;
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

    public static void main(String[] args) {
        Traversal traversal = new Traversal();
        List<String> ans = Arrays.asList(
                "cat", "cats", "and", "sand", "dog");
        ListNode root = new ListNode(1);
        ListNode prev = root;
        for (int i = 2; i <= 4; i++) {
            ListNode node = new ListNode(i);
            prev.next = node;
            prev = node;
        }
        traversal.reorderList(root);
    }

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
        if (root == null) {
            return null;
        }
        Node p = root;
        while (p.left != null) {
            Node nextLevel = p.left;

            while (p != null) {
                p.left.next = p.right;

                if (p.next != null) {
                    p.right.next = p.next.left;
                }
                p = p.next;
            }

            p = nextLevel;
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

    /**
     * 125. Valid Palindrome
     *
     * @param s
     * @return
     */
    public boolean isPalindrome(String s) {
        if (s == null) {
            return true;
        }
        s = s.trim();
        if (s.isEmpty()) {
            return true;
        }
        int left = 0;

        int right = s.length() - 1;
        while (left < right) {
            while (left < right && !Character.isLetterOrDigit(s.charAt(left))) {
                left++;
            }
            while (left < right && !Character.isLetterOrDigit(s.charAt(right))) {
                right--;
            }
            if (Character.toLowerCase(s.charAt(left)) == Character.toLowerCase(s.charAt(right))) {
                left++;
                right--;
            } else {
                return false;
            }
        }
        return true;
    }

    /**
     * 129. Sum Root to Leaf Numbers
     *
     * @param root
     * @return
     */
    public int sumNumbers(TreeNode root) {
        if (root == null) {
            return 0;
        }
        if (root.left == null && root.right == null) {
            return root.val;
        }
        return this.sumNumbers(root.left, root.val) + this.sumNumbers(root.right, root.val);
    }

    private int sumNumbers(TreeNode root, int val) {
        if (root == null) {
            return 0;
        }
        if (root.left == null && root.right == null) {
            return val * 10 + root.val;
        }
        return this.sumNumbers(root.left, val * 10 + root.val)
                + this.sumNumbers(root.right, val * 10 + root.val);
    }

    /**
     * 138. Copy List with Random Pointer
     *
     * @param head
     * @return
     */
    public Node copyRandomList(Node head) {
        if (head == null) {
            return null;
        }
        Node current = head;
        while (current != null) {
            Node tmp = current.next;

            Node next = new Node(current.val, current.next, null);

            current.next = next;

            current = tmp;
        }
        current = head;
        while (current != null) {
            Node random = current.random;

            if (random != null) {
                current.next.random = random.next;
            }
            current = current.next.next;
        }

        current = head;

        Node copyHead = head.next;
        while (current.next != null) {
            Node tmp = current.next;

            current.next = tmp.next;

            current = tmp;
        }
        return copyHead;
    }


    /**
     * todo
     * 143. Reorder List
     *
     * @param head
     */
    public void reorderList(ListNode head) {
        if (head == null || head.next == null) {
            return;
        }
        ListNode fast = head;

        ListNode slow = head;

        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }

        ListNode middle = slow;

        /**
         * 根据LeetCode
         */
        ListNode current = slow.next;


        ListNode prev = this.reverseListNode(current, null);


        slow.next = prev;


        slow = head;


        fast = middle.next;


        while (slow != middle) {
            middle.next = fast.next;

            fast.next = slow.next;

            slow.next = fast;


            slow = fast.next;

            fast = middle.next;

        }
    }

    private ListNode reverseListNode(ListNode start, ListNode end) {
        if (start == end) {
            return null;
        }
        ListNode prev = null;
        while (start != end) {
            ListNode tmp = start.next;
            start.next = prev;
            prev = start;
            start = tmp;
        }
        return prev;
    }


    /**
     * 160. Intersection of Two Linked Lists
     *
     * @param headA
     * @param headB
     * @return
     */
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if (headA == null || headB == null) {
            return null;
        }
        ListNode p1 = headA;

        ListNode p2 = headB;

        while (p1 != p2) {

            p1 = p1 == null ? headB : p1.next;
            p2 = p2 == null ? headB : p2.next;
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
            return -1;
        }
        String[] split1 = version1.split("\\.");
        String[] split2 = version2.split("\\.");
        int index1 = 0;
        int index2 = 0;
        while (index1 < split1.length || index2 < split2.length) {
            Integer value1 = index1 == split1.length ? 0 : Integer.parseInt(split1[index1++]);
            Integer value2 = index2 == split2.length ? 0 : Integer.parseInt(split2[index2++]);

            if (!value1.equals(value2)) {
                return value1.compareTo(value2);
            }
        }
        return 0;
    }


    // ---------- 深度优先遍历DFS---------//

    /**
     * 139. Word Break
     *
     * @param s
     * @param wordDict
     * @return
     */
    public boolean wordBreak(String s, List<String> wordDict) {
        if (s == null) {
            return false;
        }
        if (wordDict.contains(s) || s.isEmpty()) {
            return true;
        }
        HashMap<String, Boolean> notIncluded = new HashMap<>();
        return this.wordBreakDFS(notIncluded, s, wordDict);
    }

    private boolean wordBreakDFS(HashMap<String, Boolean> notIncluded, String s, List<String> wordDict) {
        if (notIncluded.containsKey(s)) {
            return false;
        }
        if (s.isEmpty()) {
            return true;
        }
        for (String word : wordDict) {
            if (s.startsWith(word) && this.wordBreakDFS(notIncluded, s.substring(word.length()), wordDict)) {
                return true;
            }
        }
        notIncluded.put(s, false);
        return false;
    }

    /**
     * 140. Word Break II
     *
     * @param s
     * @param wordDict
     * @return
     */
    public List<String> wordBreakII(String s, List<String> wordDict) {
        if (s == null || wordDict == null) {
            return new ArrayList<>();
        }
        HashMap<String, List<String>> map = new HashMap<>();
        return this.wordBreakIIDFS(map, s, wordDict);
    }

    private List<String> wordBreakIIDFS(HashMap<String, List<String>> map, String s, List<String> wordDict) {
        if (map.containsKey(s)) {
            return map.get(s);
        }
        List<String> ans = new ArrayList<>();
        if (s.isEmpty()) {
            ans.add("");
            return ans;
        }
        for (String word : wordDict) {
            if (s.startsWith(word)) {
                List<String> list = this.wordBreakIIDFS(map, s.substring(word.length()), wordDict);
                for (String val : list) {
                    ans.add(word + (val.isEmpty() ? "" : " ") + val);
                }
            }
        }
        return ans;
    }

    /**
     * 130. Surrounded Regions
     *
     * @param board
     */
    public void solve(char[][] board) {
        if (board == null || board.length == 0) {
            return;
        }
        int row = board.length;
        int column = board[0].length;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                boolean isEdge = i == 0 || i == row - 1 || j == 0 || j == column - 1;
                if (board[i][j] == 'O' && isEdge) {
                    this.solveBoard(i, j, board);
                }
            }
        }
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (board[i][j] == 'o') {
                    board[i][j] = 'O';
                } else if (board[i][j] == 'O') {
                    board[i][j] = 'X';
                }
            }
        }
    }

    private void solveBoard(int i, int j, char[][] board) {
        if (i < 0 || i >= board.length || j < 0 || j >= board.length || board[i][j] != 'O') {
            return;
        }
        board[i][j] = 'o';
        this.solveBoard(i - 1, j, board);

        this.solveBoard(i + 1, j, board);

        this.solveBoard(i, j - 1, board);

        this.solveBoard(i, j + 1, board);
    }


}

