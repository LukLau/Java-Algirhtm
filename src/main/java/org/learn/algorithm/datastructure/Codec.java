package org.learn.algorithm.datastructure;

import java.util.LinkedList;

/**
 * 297. Serialize and Deserialize Binary Tree
 *
 * @author luk
 * @date 2021/4/25
 */
public class Codec {

    private int index = -1;


    String Serialize(TreeNode root) {
        StringBuilder builder = new StringBuilder();
        internalSerialize(root, builder);
        return builder.toString();
    }

    private void internalSerialize(TreeNode root, StringBuilder builder) {
        if (root == null) {
            builder.append("#,");
            return;
        }
        builder.append(root.val + ",");
        internalSerialize(root.left, builder);
        internalSerialize(root.right, builder);
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
        return Deserialize(linkedList);


    }

    private TreeNode Deserialize(LinkedList<String> linkedList) {
        if (linkedList.isEmpty()) {
            return null;
        }
        String poll = linkedList.poll();
        if ("#".equals(poll)) {
            return null;
        }
        TreeNode root = new TreeNode(Integer.parseInt(poll));
        root.left = Deserialize(linkedList);
        root.right = Deserialize(linkedList);
        return root;
    }


}
