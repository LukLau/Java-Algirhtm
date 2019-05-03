package org.dora.algorithm.solution;

/**
 * @author liulu
 * @date 2019-03-16
 */
public class WordDictionary {
    TrieNode root;

    /**
     * Initialize your data structure here.
     */
    public WordDictionary() {
        root = new TrieNode();
    }

    /**
     * Adds a word into the data structure.
     */
    public void addWord(String word) {
        TrieNode p = root;
        for (int i = 0; i < word.length(); i++) {
            int index = word.charAt(i) - 'a';
            if (p.nodes[index] == null) {
                p.nodes[index] = new TrieNode();
            }
            p = p.nodes[index];
        }
        p.word = word;

    }

    /**
     * Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter.
     */
    public boolean search(String word) {
        if (word == null) {
            return false;
        }
        return this.dfs(0, word, root);
    }

    private boolean dfs(int start, String word, TrieNode root) {
        if (start == word.length()) {
            return !root.word.equals("");
        }
        if (word.charAt(start) != '.') {
            return root.nodes[word.charAt(start) - 'a'] != null && this.dfs(start + 1, word, root.nodes[word.charAt(start) - 'a']);
        } else {
            for (int i = 0; i < root.nodes.length; i++) {
                if (root.nodes[i] != null && this.dfs(start + 1, word, root.nodes[i])) {
                    return true;
                }
            }
        }
        return false;
    }


    class TrieNode {
        private TrieNode[] nodes;
        private String word;

        TrieNode() {
            nodes = new TrieNode[26];
            word = "";
        }
    }


}
