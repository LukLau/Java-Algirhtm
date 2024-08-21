package org.dora.algorithm.leetcode;

import org.dora.algorithm.datastructe.TreeNode;

/**
 * date 2024年04月21日
 * */
public class WordDirectory {

    private Trie trie;


    public WordDirectory() {
        trie = new Trie();

    }

    public void addWord(String word) {
        trie.insert(word);
    }

    public boolean search(String word) {
        if (word == null || word.isEmpty()) {
            return true;
        }
        return trie.searchii(word);
    }

}
