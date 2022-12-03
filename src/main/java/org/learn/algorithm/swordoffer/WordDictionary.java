package org.learn.algorithm.swordoffer;

import org.apache.logging.log4j.ThreadContext;

import java.util.HashMap;
import java.util.Hashtable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.AbstractQueuedSynchronizer;
import java.util.concurrent.locks.ReentrantLock;

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
