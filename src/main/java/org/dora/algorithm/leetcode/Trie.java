package org.dora.algorithm.leetcode;

/**
 * date 2024年04月21日
 */
public class Trie {

    private final TrieNode root;

    public Trie() {
        root = new TrieNode();
    }

    public TrieNode[] getTrieNode() {
        return root.next;
    }

    public void insert(String word) {
        TrieNode p = root;

        for (char tmp : word.toCharArray()) {
            if (p.next[tmp - 'a'] == null) {
                p.next[tmp - 'a'] = new TrieNode();
            }
            p = p.next[tmp - 'a'];
        }
        p.word = word;
    }

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

    public boolean startsWith(String prefix) {
        if (prefix == null || prefix.isEmpty()) {
            return true;
        }
        TrieNode p = root;
        char[] words = prefix.toCharArray();

        for (char tmp : words) {
            if (p.next[tmp - 'a'] == null) {
                return false;
            }
            p = p.next[tmp - 'a'];
        }
        return true;
    }

    public static void main(String[] args) {
        Trie trie = new Trie();
        trie.insert("apple");
        trie.search("apple");
    }

    /**
     * todo
     * @param word
     * @return
     */
    public boolean searchii(String word) {
        if (word == null || word.isEmpty()) {
            return true;
        }
        return false;
    }


    private boolean internalSearch(TrieNode p, int i, char[] words) {
        if (i == words.length) {
            return true;
        }
        if (words[i] == '.') {
            return internalSearch(p, i + 1, words);
        } else {
            for (TrieNode trieNode : p.next) {

            }

        }
        return false;
    }


    static class TrieNode {
        public String word;
        public TrieNode[] next;

        public TrieNode() {
            next = new TrieNode[26];
        }
    }
}
