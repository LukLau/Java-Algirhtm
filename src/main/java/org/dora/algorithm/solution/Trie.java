package org.dora.algorithm.solution;

/**
 * @author liulu
 * @date 2019-03-16
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
        p.word = word;
    }

    /**
     * Returns if the word is in the trie.
     */
    public boolean search(String word) {
        TrieNode p = root;
        for (int i = 0; i < word.length(); i++) {
            int index = word.charAt(i) - 'a';
            if (p.nodes[index] == null) {
                return false;
            }
            p = p.nodes[index];
        }
        return word.equals(p.word);
    }

    /**
     * Returns if there is any word in the trie that starts with the given prefix.
     */
    public boolean startsWith(String prefix) {
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
        private String word;

        public TrieNode() {
            nodes = new TrieNode[26];
            word = "";
        }
    }
}
