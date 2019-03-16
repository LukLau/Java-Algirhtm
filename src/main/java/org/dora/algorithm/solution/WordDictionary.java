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

    public static void main(String[] args) {
        WordDictionary wordDictionary = new WordDictionary();
        wordDictionary.addWord("bab");
        wordDictionary.search("..b");
    }

    /**
     * Adds a word into the data structure.
     */
    public void addWord(String word) {
        TrieNode p = root;
        for (int i = 0; i < word.length(); i++) {
            if (p.nodes[word.charAt(i) - 'a'] == null) {
                p.nodes[word.charAt(i) - 'a'] = new TrieNode();
            }
            p = p.nodes[word.charAt(i) - 'a'];
        }
        p.item = word;
    }

    /**
     * Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter.
     */
    public boolean search(String word) {
        return match(word.toCharArray(), 0, root);
    }

    private boolean match(char[] chars, int k, TrieNode root) {
        if (k == chars.length) {
            return !root.item.equals("");
        }
        if (chars[k] != '.') {
            return root.nodes[chars[k] - 'a'] != null && match(chars, k + 1, root.nodes[chars[k] - 'a']);
        } else {
            for (int i = 0; i < root.nodes.length; i++) {
                if (root.nodes[i] != null) {
                    if (match(chars, k + 1, root.nodes[i])) {
                        return true;
                    }
                }
            }
            return false;
        }
    }

    class TrieNode {
        String item = "";
        TrieNode[] nodes = new TrieNode[26];

        public TrieNode() {
        }
    }
}
