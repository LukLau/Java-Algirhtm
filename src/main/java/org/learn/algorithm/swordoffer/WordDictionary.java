package org.learn.algorithm.swordoffer;

/**
 * @author luk
 * @date 2021/8/1
 */
public class WordDictionary {

    private final Trie trie;

    /**
     * Initialize your data structure here.
     */
    public WordDictionary() {
        trie = new Trie();
    }

    public void addWord(String word) {
        trie.insert(word);
    }

    public boolean search(String word) {
        if (word == null || word.isEmpty()) {
            return false;
        }
        return trie.searchV2(word);
    }
}
