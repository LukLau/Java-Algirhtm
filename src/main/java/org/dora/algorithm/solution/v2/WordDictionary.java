package org.dora.algorithm.solution.v2;

/**
 * todo 忘掉其定义
 * @author dora
 * @date 2019/10/10
 */
public class WordDictionary {

    private Trie trie;

    /**
     * Initialize your data structure here.
     */
    public WordDictionary() {
        trie = new Trie();
    }

    /**
     * Adds a word into the data structure.
     */
    public void addWord(String word) {
        trie.insert(word);
    }

    /**
     * Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter.
     */
    public boolean search(String word) {
        if (word == null || word.isEmpty()) {
            return false;
        }
        return false;
    }

    private boolean intervalSearch(int index, String word) {
        if (index == word.length()) {
            return true;
        }
        return false;
    }
}
