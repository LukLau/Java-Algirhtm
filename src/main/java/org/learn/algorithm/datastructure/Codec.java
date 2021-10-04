package org.learn.algorithm.datastructure;

import java.util.LinkedList;

/**
 * 297. Serialize and Deserialize Binary Tree
 *
 * @author luk
 * @date 2021/4/25
 */
public class Codec {


    String Serialize(TreeNode root) {
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

    TreeNode Deserialize(String str) {
        if (str == null || str.isEmpty()) {
            return null;
        }
        String[] words = str.split(",");
        LinkedList<String> linkedList = new LinkedList<>();
        for (String word : words) {
            linkedList.offer(word);
        }
        return intervalDeserialize(linkedList);
    }

    private TreeNode intervalDeserialize(LinkedList<String> linkedList) {
        if (linkedList.isEmpty()) {
            return null;
        }
        String poll = linkedList.poll();
        if ("#".equals(poll)) {
            return null;
        }
        TreeNode root = new TreeNode(Integer.parseInt(poll));
        root.left = intervalDeserialize(linkedList);
        root.right = intervalDeserialize(linkedList);
        return root;
    }

}
