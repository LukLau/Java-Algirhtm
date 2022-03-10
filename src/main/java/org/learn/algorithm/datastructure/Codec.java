package org.learn.algorithm.datastructure;

/**
 * 297. Serialize and Deserialize Binary Tree
 *
 * @author luk
 * @date 2021/4/25
 */
public class Codec {

    private int index = -1;

    public String Serialize(TreeNode root) {
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

    public TreeNode Deserialize(String str) {
        if (str == null || str.isEmpty()) {
            return null;
        }
        String[] words = str.split(",");
        return internalDeserialize(words);
    }

    private TreeNode internalDeserialize(String[] words) {
        index++;
        if (index == words.length) {
            return null;
        }
        if ("#".equals(words[index])) {
            return null;
        }
        TreeNode root = new TreeNode(Integer.parseInt(words[index]));
        root.left = internalDeserialize(words);
        root.right = internalDeserialize(words);
        return root;
    }

}
