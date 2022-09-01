package org.learn.algorithm.leetcode;

import lombok.Value;
import org.learn.algorithm.datastructure.DirectedGraphNode;
import org.learn.algorithm.datastructure.Node;
import org.learn.algorithm.datastructure.UndirectedGraphNode;

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

    public static void main(String[] args) {
        GraphSolution graphSolution = new GraphSolution();

//        Node root = new Node(1);
//
//        List<Node> neighbors = new ArrayList<>();
//
//        Node node2 = new Node(2);
//        Node node4 = new Node(4);
//
//
//        neighbors.add(node2);
//        neighbors.add(node4);
//
//        root.neighbors = neighbors;
//
//
//        List<Node> neighbors2 = new ArrayList<>();
//
//        neighbors2.add(root);
//        neighbors2.add(node4);
//
//        node2.neighbors = neighbors2;
//
//
//        List<Node> neighbors4 = new ArrayList<>();
//
//        neighbors4.add(root);
//        neighbors4.add(node2);
//
//        node4.neighbors = neighbors4;

        int numCourses = 2;
        int[][] matrix = new int[][]{{1, 0}, {0, 1}};

        graphSolution.canFinish(numCourses, matrix);
    }


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
        Map<Integer, Node> map = new HashMap<>();

        return internalCloneNode(map, node);
    }

    private Node internalCloneNode(Map<Integer, Node> map, Node node) {
        if (node == null) {
            return null;
        }
        if (map.containsKey(node.val)) {
            return map.get(node.val);
        }
        Node root = new Node(node.val);

        map.put(root.val, root);
        for (Node neighbor : node.neighbors) {
            root.neighbors.add(internalCloneNode(map, neighbor));
        }
        return root;
    }


    /**
     * https://www.lintcode.com/problem/137/
     *
     * @param node: A undirected graph node
     * @return: A undirected graph node
     */
    public UndirectedGraphNode cloneGraph(UndirectedGraphNode node) {
        // write your code here
        if (node == null) {
            return null;
        }
        Map<Integer, UndirectedGraphNode> map = new HashMap<>();
        return internalClone(map, node);
    }

    private UndirectedGraphNode internalClone(Map<Integer, UndirectedGraphNode> map, UndirectedGraphNode root) {
        if (root == null) {
            return null;
        }
        UndirectedGraphNode node = map.get(root.label);

        if (node != null) {
            return node;
        }
        UndirectedGraphNode undirectedGraphNode = new UndirectedGraphNode(root.label);
        map.put(undirectedGraphNode.label, undirectedGraphNode);

        for (UndirectedGraphNode neighbor : root.neighbors) {
            undirectedGraphNode.neighbors.add(internalClone(map, neighbor));
        }
        return undirectedGraphNode;
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

        List<Integer> result = new ArrayList<>();

        Map<Integer, Integer> graphDegree = getGraphDegree(prerequisites);

        LinkedList<Integer> linkedList = new LinkedList<>();

        for (int[] prerequisite : prerequisites) {
            if (graphDegree.containsKey(prerequisite[1])) {
                continue;
            }
            linkedList.offer(prerequisite[1]);
        }
        Map<Integer, Integer> convertToMap = convertMap(prerequisites);

        while (!linkedList.isEmpty()) {
            Integer poll = linkedList.poll();

            result.add(poll);


            Integer next = convertToMap.get(poll);


            Integer count = graphDegree.getOrDefault(next, 0);

            count--;

            graphDegree.put(next, count);

            if (count == 0) {
                linkedList.offer(convertToMap.get(next));
            }
        }
        return result.size() == numCourses;
    }

    private Map<Integer, Integer> convertMap(int[][] matrix) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int[] row : matrix) {
            map.put(row[0], row[1]);
        }
        return map;
    }

    private Map<Integer, Integer> getGraphDegree(int[][] graph) {
        Map<Integer, Integer> map = new HashMap<>();

        for (int[] nodes : graph) {
            Integer count = map.getOrDefault(nodes[1], 0);

            map.put(nodes[1], count + 1);
        }
        return map;
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


    /**
     * https://www.lintcode.com/problem/127/description
     *
     * @param graph: A list of Directed graph node
     * @return: Any topological order for the given graph.
     */
    public ArrayList<DirectedGraphNode> topSort(ArrayList<DirectedGraphNode> graph) {
        // write your code here
        if (graph == null || graph.isEmpty()) {
            return new ArrayList<>();
        }
        ArrayList<DirectedGraphNode> result = new ArrayList<>();
        Map<DirectedGraphNode, Integer> map = new HashMap<>();

        for (DirectedGraphNode directedGraphNode : graph) {
            internalTop(result, directedGraphNode, map);
        }
        Collections.reverse(result);
        return result;
    }

    private void internalTop(ArrayList<DirectedGraphNode> result, DirectedGraphNode node, Map<DirectedGraphNode, Integer> map) {
        if (map.computeIfAbsent(node, value -> 0) == 1) {
            return;
        }
        map.compute(node, (key, value) -> 1);
        if (node.neighbors == null || node.neighbors.isEmpty()) {
            result.add(node);
            return;
        }
        for (DirectedGraphNode neighbor : node.neighbors) {
            internalTop(result, neighbor, map);
        }
        result.add(node);
    }

    public ArrayList<DirectedGraphNode> topSortii(ArrayList<DirectedGraphNode> graph) {
        if (graph == null || graph.isEmpty()) {
            return new ArrayList<>();
        }
        Map<DirectedGraphNode, Integer> graphNodeDegree = getGraphNodeDegree(graph);

        LinkedList<DirectedGraphNode> linkedList = new LinkedList<>();

        ArrayList<DirectedGraphNode> result = new ArrayList<>();

        for (DirectedGraphNode directedGraphNode : graph) {
            if (graphNodeDegree.containsKey(directedGraphNode)) {
                continue;
            }
            linkedList.add(directedGraphNode);
        }

        while (!linkedList.isEmpty()) {
            DirectedGraphNode directedGraphNode = linkedList.poll();

            result.add(directedGraphNode);

            for (DirectedGraphNode neighbor : directedGraphNode.neighbors) {

                Integer count = graphNodeDegree.get(neighbor);

                count--;

                graphNodeDegree.put(neighbor, count);

                if (count == 0) {
                    linkedList.offer(neighbor);
                }
            }
        }
        return result;
    }

    private Map<DirectedGraphNode, Integer> getGraphNodeDegree(ArrayList<DirectedGraphNode> graph) {
        Map<DirectedGraphNode, Integer> map = new HashMap<>();
        for (DirectedGraphNode directedGraphNode : graph) {
            for (DirectedGraphNode neighbor : directedGraphNode.neighbors) {
                Integer inputEdge = map.getOrDefault(neighbor, 0);
                map.put(neighbor, inputEdge + 1);
            }
        }
        return map;
    }


}
