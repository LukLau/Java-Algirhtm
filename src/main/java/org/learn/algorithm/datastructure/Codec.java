package org.learn.algorithm.datastructure;

import java.util.LinkedList;

/**
 * 297. Serialize and Deserialize Binary Tree
 *
 * @author luk
 * @date 2021/4/25
 */
public class Codec {

    // Encodes a tree to a single string.

    public String serialize(TreeNode root) {
        if (root == null) {
            return "#,";
        }
        StringBuilder builder = new StringBuilder();
        dfs(builder, root);
        return builder.toString();

    }

    private void dfs(StringBuilder builder, TreeNode root) {
        if (root == null) {
            builder.append("#,");
            return;
        }
        builder.append(root.val).append(",");
        dfs(builder, root.left);
        dfs(builder, root.right);
    }

    // Decodes your encoded data to tree.

    public TreeNode deserialize(String data) {
        if (data == null || data.isEmpty()) {
            return null;
        }
        String[] words = data.split(",");
        LinkedList<String> linkedList = new LinkedList<>();
        for (String word : words) {
            linkedList.offer(word);
        }
        return internalDeserialize(linkedList);

    }

    private TreeNode internalDeserialize(LinkedList<String> linkedList) {
        if (linkedList.isEmpty()) {
            return null;
        }
        String poll = linkedList.poll();

        if ("#".equals(poll)) {
            return null;
        }
        TreeNode root = new TreeNode(Integer.parseInt(poll));
        root.left = internalDeserialize(linkedList);
        root.right = internalDeserialize(linkedList);
        return root;
    }


}
