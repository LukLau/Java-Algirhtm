package org.learn.algorithm.datastructure;

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
        return trie.searchV2(word);
    }


    public static void main(String[] args) {
        WordDictionary dictionary = new WordDictionary();

        dictionary.addWord("a");
        dictionary.search(".a");
    }


    public boolean startWith(String s) {
        return trie.startsWith(s);
    }
}
