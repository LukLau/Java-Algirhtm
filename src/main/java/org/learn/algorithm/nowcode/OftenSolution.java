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


}
