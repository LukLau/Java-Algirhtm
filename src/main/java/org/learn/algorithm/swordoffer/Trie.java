package org.learn.algorithm.swordoffer;

/**
 * 字典树
 *
 * @author luk
 * @date 2021/8/1
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
        if (word == null || word.isEmpty()) {
            return;
        }
        TrieNode p = root;
        char[] words = word.toCharArray();
        for (char tmp : words) {
            int index = tmp - 'a';
            if (p.next[index] == null) {
                p.next[index] = new TrieNode();
            }
            p = p.next[index];
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
            int index = tmp - 'a';
            if (p.next[index] == null) {
                return false;
            }
            p = p.next[index];
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
        int len = prefix.length();
        for (int i = 0; i < len; i++) {
            int index = prefix.charAt(i) - 'a';
            if (p.next[index] == null) {
                return false;
            }
            p = p.next[index];
        }
        return true;
    }


    public static void main(String[] args) {
        Trie trie = new Trie();
        trie.insert("apple");
    }

    public boolean searchV2(String word) {
        if (word == null || word.isEmpty()) {
            return false;
        }
        TrieNode p = root;
        return intervalSearchV2(p, 0, word.toCharArray());
    }

    private boolean intervalSearchV2(TrieNode root, int start, char[] words) {
        if (start == words.length) {
            return root.word != null;
        }
        if (words[start] != '.') {
            int index = words[start] - 'a';
            if (root.next[index] == null) {
                return false;
            }
            return intervalSearchV2(root.next[index], start + 1, words);
        }
        for (TrieNode trieNode : root.next) {
            if (trieNode != null && intervalSearchV2(trieNode, start + 1, words)) {
                return true;
            }
        }
        return false;
    }
}


class TrieNode {
    protected String word;

    protected TrieNode[] next;


    public TrieNode() {
        next = new TrieNode[26];
    }

    public TrieNode(String word) {
        this.word = word;
        next = new TrieNode[26];
    }
}
