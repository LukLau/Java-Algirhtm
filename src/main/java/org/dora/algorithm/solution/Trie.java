package org.dora.algorithm.solution;

/**
 * @author liulu
 * @date 2019-03-16
 */
public class Trie {
    private TrieNode root;
    /** Initialize your data structure here. */
    public Trie() {
        root = new TrieNode();
    }

    /** Inserts a word into the trie. */
    public void insert(String word) {
        TrieNode p = root;
        for (int i = 0; i < word.length(); i++) {
            if (p.words[word.charAt(i) - 'a'] == null) {
                p.words[word.charAt(i) - 'a'] = new TrieNode();
            }
            p = p.words[word.charAt(i) - 'a'];
        }
        p.hasNext = true;
    }

    /** Returns if the word is in the trie. */
    public boolean search(String word) {
        TrieNode p = root;
        for (int i = 0; i < word.length(); i++) {
            if (p.words[word.charAt(i) - 'a'] == null) {
                return false;
            }
            p = p.words[word.charAt(i) - 'a'];
        }
        return p.hasNext;
    }


    /** Returns if there is any word in the trie that starts with the given prefix. */
    public boolean startsWith(String prefix) {
        TrieNode p = root;
        for (int i = 0; i < prefix.length(); i++) {
            if (p.words[prefix.charAt(i) - 'a'] == null) {
                return false;
            }
            p = p.words[prefix.charAt(i) - 'a'];
        }
        return true;
    }

    class TrieNode {
        private boolean hasNext = false;
        private TrieNode[] words;

        public TrieNode() {
            words = new TrieNode[26];
        }
    }
}
