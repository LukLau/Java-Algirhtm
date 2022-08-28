package org.learn.algorithm.datastructure;

import java.util.ArrayList;
import java.util.List;

public class UndirectedGraphNode {

    public int label;
    public List<UndirectedGraphNode> neighbors;

    public UndirectedGraphNode(int x) {
        label = x;
        neighbors = new ArrayList<UndirectedGraphNode>();
    }


}
