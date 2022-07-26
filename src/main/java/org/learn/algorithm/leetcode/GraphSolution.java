package org.learn.algorithm.leetcode;

import org.learn.algorithm.datastructure.Node;

import javax.print.attribute.PrintRequestAttribute;
import javax.swing.*;
import java.util.*;

/**
 * 图相关
 *
 * @author luk
 * @date 2021/4/12
 */
public class GraphSolution {


    /**
     * 133. Clone Graph
     *
     * @param node
     * @return
     */
    public Node cloneGraph(Node node) {
        if (node == null) {
            return null;
        }
        Set<Integer> used = new HashSet<>();

        LinkedList<Node> linkedList = new LinkedList<>();

        linkedList.offer(node);

        Node root = null;
        while (!linkedList.isEmpty()) {
            Node prev = linkedList.poll();

            if (used.contains(prev.val)) {
                continue;
            }
            Node expectedNode = new Node(prev.val);

            if (root == null) {
                root = expectedNode;
            }
            used.add(expectedNode.val);

            List<Node> prevNeighbors = prev.neighbors;

            List<Node> expectedNeighbors = new ArrayList<>();

            if (prevNeighbors != null && !prevNeighbors.isEmpty()) {
                for (Node prevNeighbor : prevNeighbors) {
                    if (prevNeighbor != null) {
                        Node expectedNeighbor = new Node(prevNeighbor.val);
                        expectedNeighbors.add(expectedNeighbor);
                        linkedList.offer(prevNeighbor);
                    }
                }
            }
            expectedNode.neighbors = expectedNeighbors;
        }
        return root;
    }


    /**
     * todo
     * 207. Course Schedule
     *
     * @param numCourses
     * @param prerequisites
     * @return
     */
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        if (prerequisites == null || prerequisites.length == 0) {
            return false;
        }
        List<Integer> vertex = new ArrayList<>();
        List<List<Integer>> edges = new ArrayList<>();

        for (int i = 0; i < prerequisites.length; i++) {


        }


        return false;
    }

    /**
     * todo
     * 210. Course Schedule II
     *
     * @param numCourses
     * @param prerequisites
     * @return
     */
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        return null;
    }


    /**
     * todo
     * 310. Minimum Height Trees
     *
     * @param n
     * @param edges
     * @return
     */
    public List<Integer> findMinHeightTrees(int n, int[][] edges) {
        return null;
    }


    /**
     * todo
     * NC159 最小生成树
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * <p>
     * 返回最小的花费代价使得这n户人家连接起来
     *
     * @param n    int n户人家的村庄
     * @param m    int m条路
     * @param cost int二维数组 一维3个参数，表示连接1个村庄到另外1个村庄的花费的代价
     * @return int
     */
    public int miniSpanningTree(int n, int m, int[][] cost) {
        // write code here
        return 0;
    }

    /**
     * NC158 单源最短路
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * @param n     int 顶点数
     * @param m     int 边数
     * @param graph int二维数组 一维3个数据，表示顶点到另外一个顶点的边长度是多少​
     * @return int
     */
    public int findShortestPath(int n, int m, int[][] graph) {
        // write code here
        return -1;
    }


}
