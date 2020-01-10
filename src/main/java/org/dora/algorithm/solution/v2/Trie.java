package org.dora.algorithm.solution.v2;

/**
 * @author dora
 * @date 2019/10/9
 */
public class Trie {

    private TrieNode root;

    /**
     * Initialize your data structure here.
     */
    public Trie() {
        root = new TrieNode();
    }

    /**
     * Inserts a word into the trie.
     */
    public void insert(String word) {
        TrieNode p = root;
        for (int i = 0; i < word.length(); i++) {
            int index = word.charAt(i) - 'a';
            if (p.nodes[index] == null) {
                p.nodes[index] = new TrieNode();
            }
            p = p.nodes[index];
        }
        p.value = word;
    }

    /**
     * Returns if the word is in the trie.
     */
    public boolean search(String word) {
        if (word == null || word.isEmpty()) {
            return false;
        }
        TrieNode p = root;
        for (int i = 0; i < word.length(); i++) {
            int index = word.charAt(i) - 'a';
            if (p.nodes[index] == null) {
                return false;
            }
            p = p.nodes[index];
        }
        return p.value.equals(word);
    }

    /**
     * Returns if there is any word in the trie that starts with the given prefix.
     */
    public boolean startsWith(String prefix) {
        if (prefix == null) {
            return false;
        }
        TrieNode p = root;
        for (int i = 0; i < prefix.length(); i++) {
            int index = prefix.charAt(i) - 'a';
            if (p.nodes[index] == null) {
                return false;
            }
            p = p.nodes[index];
        }
        return true;
    }


    class TrieNode {
        private TrieNode[] nodes;
        private String value;

        public TrieNode() {
            nodes = new TrieNode[26];
            value = "";
        }
    }
}
