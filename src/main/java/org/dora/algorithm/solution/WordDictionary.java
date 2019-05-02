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
    }

    /**
     * Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter.
     */
    public boolean search(String word) {

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
