package org.learn.algorithm.datastructure;

/**
 * 字典树
 *
 * @author luk
 * @date 2021/4/14
 */
public class Trie {

    public Trie() {

    }

    public void insert(String word) {

    }

    public boolean search(String word) {

    }

    public boolean startsWith(String prefix) {

    }

    static class TrieNode {
        public char[] next;
        public String words;

        public TrieNode() {
            next = new char[26];
        }
    }

}

