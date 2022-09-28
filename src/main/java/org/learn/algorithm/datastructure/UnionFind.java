package org.learn.algorithm.datastructure;


public class UnionFind {
    private final int[] parents;

    private int count;

    public UnionFind(int totalNodes) {
        parents = new int[totalNodes];
        for (int i = 0; i < totalNodes; i++) {
            parents[i] = i;
        }
    }

    public int find(int node) {
        if (node == parents[node]) {
            return node;
        }
        parents[node] = find(parents[node]);

        return parents[node];
    }

    public void connect(int nodeA, int nodeB) {
        int rootA = find(nodeA);
        int rootB = find(nodeB);
        if (rootA != rootB) {
            parents[rootA] = rootB;
            count--;
        }
    }

    public int query() {
        return count;
    }

    public void setCount(int total) {
        count = total;
    }


}
