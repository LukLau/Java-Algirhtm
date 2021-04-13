package org.learn.algorithm.datastructure;

import javax.xml.crypto.dsig.keyinfo.RetrievalMethod;
import javax.xml.stream.FactoryConfigurationError;

/**
 * 字典
 *
 * @author luk
 * @date 2021/4/14
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
        if (word == null || word.isEmpty()) {
            return;
        }
        trie.insert(word);
    }

    public boolean search(String word) {
        if (word == null) {
            return false;
        }
        return false;
    }

}
