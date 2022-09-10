package org.learn.algorithm.leetcode;

import org.learn.algorithm.datastructure.DirectedGraphNode;

import java.util.*;

public class VipGraph {
    List<List<Integer>> edges;
    int[] indeg;

    /**
     * 269. Alien Dictionary
     *
     * @param words: a list of words
     * @return: a string which is correct order
     */
    public String alienOrder(String[] words) {
        // Write your code here
        if (words == null || words.length == 0) {
            return "";
        }
        return "";
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
            internalTopSortDFS(result, map, directedGraphNode);
        }
        Collections.reverse(result);
        return result;
    }

    private void internalTopSortDFS(ArrayList<DirectedGraphNode> result, Map<DirectedGraphNode, Integer> map, DirectedGraphNode graphNode) {
        if (map.containsKey(graphNode)) {
            return;
        }
        map.put(graphNode, 1);
        for (DirectedGraphNode neighbor : graphNode.neighbors) {
            internalTopSortDFS(result, map, neighbor);
        }
        result.add(graphNode);
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

    public boolean canFinishTest(int numCourses, int[][] prerequisites) {
        edges = new ArrayList<List<Integer>>();
        for (int i = 0; i < numCourses; ++i) {
            edges.add(new ArrayList<Integer>());
        }
        indeg = new int[numCourses];
        for (int[] info : prerequisites) {
            edges.get(info[1]).add(info[0]);
            ++indeg[info[0]];
        }

        Queue<Integer> queue = new LinkedList<Integer>();
        for (int i = 0; i < numCourses; ++i) {
            if (indeg[i] == 0) {
                queue.offer(i);
            }
        }

        int visited = 0;
        while (!queue.isEmpty()) {
            ++visited;
            int u = queue.poll();
            for (int v : edges.get(u)) {
                --indeg[v];
                if (indeg[v] == 0) {
                    queue.offer(v);
                }
            }
        }

        return visited == numCourses;
    }

    /**
     * 261. Graph Valid Tree
     * https://www.lintcode.com/problem/178/
     *
     * @param n:     An integer
     * @param edges: a list of undirected edges
     * @return: true if it's a valid tree, or false
     */
    public boolean validTree(int n, int[][] edges) {
        // write your code here
        if (edges == null || edges.length == 0) {
            return false;
        }
        boolean[] used = new boolean[n];


        Map<Integer, Integer> graphDegree = getGraphDegree(edges);

        LinkedList<Integer> linkedList = new LinkedList<>();

        for (int[] edge : edges) {
            int outEdge = edge[0];
            int inputDegree = graphDegree.getOrDefault(outEdge, 0);
            if (inputDegree == 0) {
                linkedList.offer(inputDegree);
            }
        }
        List<Integer> result = new ArrayList<>();

        while (!linkedList.isEmpty()) {
            Integer poll = linkedList.poll();

            if (used[poll]) {
                return false;
            }
        }

        return false;
    }


    private Map<Integer, Integer> getGraphDegree(int[][] graph) {
        Map<Integer, Integer> map = new HashMap<>();

        for (int[] nodes : graph) {
            Integer count = map.getOrDefault(nodes[0], 0);

            map.put(nodes[0], count + 1);
        }
        return map;
    }
}
