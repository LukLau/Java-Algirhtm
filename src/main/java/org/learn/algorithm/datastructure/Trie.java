package org.learn.algorithm.datastructure;

import com.sun.java.swing.action.NextAction;

/**
 * 字典树
 *
 * @author luk
 * @date 2021/4/14
 */
public class Trie {

    private final TrieNode root;

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
        if (word == null || word.isEmpty()) {
            return;
        }
        char[] words = word.toCharArray();

        TrieNode p = root;
        for (char tmp : words) {
            if (p.next[tmp - 'a'] == null) {
                p.next[tmp - 'a'] = new TrieNode();
            }
            p = p.next[tmp - 'a'];
        }
        p.word = word;
    }

    /**
     * Returns if the word is in the trie.
     */
    public boolean search(String word) {
        if (word == null || word.isEmpty()) {
            return false;
        }
        TrieNode p = root;
        char[] words = word.toCharArray();
        for (char tmp : words) {
            if (p.next[tmp - 'a'] == null) {
                return false;
            }
            p = p.next[tmp - 'a'];
        }
        return word.equals(p.word);
    }

    /**
     * Returns if there is any word in the trie that starts with the given prefix.
     */
    public boolean startsWith(String prefix) {
        if (prefix == null || prefix.isEmpty()) {
            return false;
        }
        TrieNode p = root;
        char[] prefixWords = prefix.toCharArray();
        for (char prefixWord : prefixWords) {
            if (p.next[prefixWord - 'a'] == null) {
                return false;
            }
            p = p.next[prefixWord - 'a'];
        }
        return true;
    }


    static class TrieNode {
        private String word;
        private final TrieNode[] next;

        public TrieNode() {
            next = new TrieNode[26];
        }
    }

}
