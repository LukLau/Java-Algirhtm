package org.learn.algorithm.leetcode;

import org.learn.algorithm.datastructure.DirectedGraphNode;
import org.learn.algorithm.datastructure.Node;
import org.learn.algorithm.datastructure.UndirectedGraphNode;

import java.util.*;

/**
 * 图相关
 *
 * @author luk
 * @date 2021/4/12
 */
public class GraphSolution {

    List<List<Integer>> edges;
    int[] indeg;

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

        int numCourses = 4;
//        int[][] matrix = new int[][]{{1, 0}, {0, 1}};

        int[][] matrix = new int[][]{{1, 0}, {2, 0}, {3, 1}, {3, 2}};

        matrix = new int[][]{{0, 1}, {1, 2}, {2, 3}, {1, 3}, {1, 4}};

        matrix = new int[][]{{0, 1}, {1, 2}, {3, 2}, {4, 3}, {4, 5}, {5, 6}, {6, 7}};


//        graphSolution.validTreeii(5, matrix);

        graphSolution.validTreeiii(8, matrix);
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
        if (prerequisites == null) {
            return false;
        }
        List<List<Integer>> result = new ArrayList<>();

        for (int i = 0; i < numCourses; i++) {
            List<Integer> tmp = new ArrayList<>();
            result.add(tmp);
        }
        for (int[] prerequisite : prerequisites) {
            List<Integer> tmp = result.get(prerequisite[1]);
            tmp.add(prerequisite[0]);
        }
        Map<Integer, Integer> graphDegree = getGraphDegree(prerequisites);

        LinkedList<Integer> linkedList = new LinkedList<>();

        for (int i = 0; i < numCourses; i++) {
            if (graphDegree.containsKey(i)) {
                continue;
            }
            linkedList.offer(i);
        }
        List<Integer> nums = new ArrayList<>();
        while (!linkedList.isEmpty()) {
            Integer poll = linkedList.poll();

            nums.add(poll);

            List<Integer> tmp = result.get(poll);

            for (Integer num : tmp) {

                int count = graphDegree.getOrDefault(num, 0);

                count--;

                graphDegree.put(num, count);

                if (count == 0) {
                    linkedList.offer(num);
                }
            }
        }
        int[] tmp = new int[nums.size()];

        for (int i = 0; i < nums.size(); i++) {
            tmp[i] = nums.get(i);
        }
//        return tmp;

        return nums.size() == numCourses;
    }

    /**
     * https://www.lintcode.com/problem/616
     * todo
     * 210. Course Schedule II
     *
     * @param numCourses
     * @param prerequisites
     * @return
     */
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        if (prerequisites == null) {
            return new int[]{};
        }


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

    private Map<Integer, Integer> getGraphDegree(int[][] graph) {
        Map<Integer, Integer> map = new HashMap<>();

        for (int[] nodes : graph) {
            Integer count = map.getOrDefault(nodes[1], 0);

            map.put(nodes[1], count + 1);
        }
        return map;
    }


    // Graph Valid Tree

    /**
     * https://www.lintcode.com/problem/178/
     *
     * @return
     */
    /**
     * @param n:     An integer
     * @param edges: a list of undirected edges
     * @return: true if it's a valid tree, or false
     */
    public boolean validTree(int n, int[][] edges) {
        // write your code here
//        if (edges == null) {
//            return false;
//        }
//        if (n == 1 && edges.length == 0) {
//            return true;
//        }
//        Map<Integer, Integer> graphDegree = getGraphDegree(edges);
//        Map<Integer, List<Integer>> nodes = new HashMap<>();
//
//        for (int[] edge : edges) {
//            List<Integer> tmp = nodes.getOrDefault(edge[0], new ArrayList<>());
//            tmp.add(edge[1]);
//            nodes.put(edge[0], tmp);
//        }
//        LinkedList<Integer> linkedList = new LinkedList<>();
//        for (Map.Entry<Integer, List<Integer>> item : nodes.entrySet()) {
//            Integer node = item.getKey();
//            Integer count = graphDegree.getOrDefault(node, 0);
//            if (count == 0) {
//                linkedList.offer(node);
//            }
//        }
//        List<Integer> result = new ArrayList<>();
//        boolean[] used = new boolean[n];
//
//        while (!linkedList.isEmpty()) {
//            Integer poll = linkedList.poll();
//            System.out.println(poll);
//            result.add(poll);
//            List<Integer> neighbors = nodes.getOrDefault(poll, new ArrayList<>());
//
//            System.out.println("node: " + poll + " neighbors:" + neighbors);
//            for (Integer neighbor : neighbors) {
////                if (result.contains(neighbor)) {
////                    continue;
////                }
//                if (used[neighbor]) {
//                    continue;
//                }
//                used[neighbor] = true;
//                Integer count = graphDegree.getOrDefault(neighbor, 0);
//
//                count--;
//                graphDegree.put(neighbor, count);
//                if (count == 0) {
//                    linkedList.offer(neighbor);
//
//                }
//            }
//        }
//        return result.size() == n;
        return false;
    }


    /**
     * 来自官方的解释
     *
     * @param n
     * @param edges
     * @return
     */
    public boolean validTreeii(int n, int[][] edges) {
        if (n == 0) {
            return false;
        }
        // 判断图是否是树依据上述的第二个条件
        if (edges.length != n - 1) {
            return false;
        }

        // set 用于去重，不包含重复元素的集合
        Map<Integer, Set<Integer>> graph = initializeGraph(n, edges);

        System.out.println("valid treeii " +  graph);

        // bfs
        // queue 里面存的是结点下标
        Queue<Integer> queue = new LinkedList<>();
        Set<Integer> hash = new HashSet<>();

        queue.offer(0);
        hash.add(0);     // hashset 没有 offer 用法
        // queue 结合 while 使用来保证遍历全部结点
        while (!queue.isEmpty()) {
            int node = queue.poll();
            // foreach 用法，neighbor 是变量名
            // graph.get(node) 对应的是一个集合
            for (Integer neighbor : graph.get(node)) {
                // hash 表用于去除重复结点，来保证队列中没有添加重复结点
                // 树只能从上往下遍历，但图没有方向，A 是 B 的相邻结点，B 也是A 的相邻结点，所以要去重
                if (hash.contains(neighbor)) {
                    continue;
                }
                hash.add(neighbor);
                queue.offer(neighbor);
            }
        }

        // 当存在 n-1 条边且有 n 个结点连通时说明图是树
        return (hash.size() == n);
    }

    // 根据结点和边初始化一张图出来
    private Map<Integer, Set<Integer>> initializeGraph(int n, int[][] edges) {
        // set 的不包含重复元素特性特别重要，set 在后边代表的两点间建立边的关系
        // 若某一点重复加了一个点两次则证明出现了环，初始化必须保证无环，算法才有意义
        Map<Integer, Set<Integer>> graph = new HashMap<>();
        for (int i = 0; i < n; i++) {
            // hashset 用于存储不重复整型对象,
            // hashmap 中的 put 方法用于关联指定值与指定键，
            // 本行代码用于创建 n 个映射
            graph.put(i, new HashSet<Integer>());
        }

        // 注意此处不是 n，n 代表结点数
        // i 循环的是边数，边数小于 n，若写成 n 则在 i = n - 1 时代码会卡住，程序超时
        for (int i = 0; i < edges.length; i++) {
            int u = edges[i][0];
            int v = edges[i][1];
            graph.get(u).add(v);
            graph.get(v).add(u);
        }
        // 在图中建立边的连接实际上就是建立两个整数间的不重复映射关系
        // get() 返回指定键映射的值,即 graph 代表 hashset 数组,
        // graph.get(v).add(u) 代表 hashset.add()
        // u 和 v 代表边的两个端点在 graph.get(u) 中 u,v  代表索引值 i，
        // u.add(v) 是指加一条 u 到 v 的边(graph 中下标为 u 的 set 加入
        // 一个值为 v 的元素)，把题目提供的输入数据中的边数组进行处理

        return graph;
    }


    public boolean validTreeiii(int n, int[][] edges) {
        if (edges == null) {
            return false;
        }
        if (edges.length != n-1) {
            return false;
        }

        Map<Integer, Set<Integer>> graph = constructUndirectedGraph(n,edges);

        System.out.println("valid treeiii " +  graph);
        LinkedList<Integer> queue = new LinkedList<>();

        Set<Integer> hash = new HashSet<>();

        queue.offer(0);
        hash.add(0);
        while (!queue.isEmpty()) {
            Integer poll = queue.poll();

            Set<Integer> neighbors = graph.getOrDefault(poll, new HashSet<>());

            for (Integer neighbor : neighbors) {
                if (hash.contains(neighbor)) {
                    continue;
                }
                hash.add(neighbor);
                queue.offer(neighbor);
            }
        }
        return hash.size() == n;
    }


    private Map<Integer, Set<Integer>> constructUndirectedGraph(int n, int[][] edges) {
        Map<Integer, Set<Integer>> graph = new HashMap<>();
        for (int i = 0; i < n; i++) {
            // hashset 用于存储不重复整型对象,
            // hashmap 中的 put 方法用于关联指定值与指定键，
            // 本行代码用于创建 n 个映射
            graph.put(i, new HashSet<>());
        }

        // 注意此处不是 n，n 代表结点数
        // i 循环的是边数，边数小于 n，若写成 n 则在 i = n - 1 时代码会卡住，程序超时
        for (int i = 0; i < edges.length; i++) {
            int u = edges[i][0];
            int v = edges[i][1];
            graph.get(u).add(v);
            graph.get(v).add(u);
        }
        return graph;
    }


}
