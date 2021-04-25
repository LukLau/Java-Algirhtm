package org.learn.algorithm.datastructure;

import java.util.Deque;
import java.util.LinkedList;

/**
 * 297. Serialize and Deserialize Binary Tree
 *
 * @author luk
 * @date 2021/4/25
 */
public class Codec {


    /**
     * Encodes a tree to a single string.
     *
     * @param root
     * @return
     */
    public String serialize(TreeNode root) {
        if (root == null) {
            return "#,";
        }
        StringBuilder builder = new StringBuilder();
        intervalSerialize(builder, root);
        return builder.toString();
    }

    private void intervalSerialize(StringBuilder builder, TreeNode root) {
        if (root == null) {
            builder.append("#,");
            return;
        }
        builder.append(root.val).append(",");
        intervalSerialize(builder, root.left);
        intervalSerialize(builder, root.right);
    }

    /**
     * Decodes your encoded data to tree.
     *
     * @param data
     * @return
     */
    public TreeNode deserialize(String data) {
        if (data == null || data.isEmpty()) {
            return null;
        }
        Deque<String> deque = new LinkedList<>();
        String[] words = data.split(",");
        for (String word : words) {
            deque.offer(word);
        }
        return intervalDeserialize(deque);
    }

    private TreeNode intervalDeserialize(Deque<String> deque) {
        if (deque.isEmpty()) {
            return null;
        }
        String poll = deque.poll();
        if ("#".equals(poll)) {
            return null;
        }
        TreeNode root = new TreeNode(Integer.parseInt(poll));
        root.left = intervalDeserialize(deque);
        root.right = intervalDeserialize(deque);
        return root;
    }
}
