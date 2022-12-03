package org.learn.algorithm.datastructure;

/**
 * 字典
 *
 * @author luk
 * @date 2021/4/14
 */
public class WordDictionary {

    private final Trie root;

    public WordDictionary() {
        root = new Trie();
    }

    public void addWord(String word) {
        if (word == null || word.isEmpty()) {
            return;
        }
        root.insert(word);
    }

    public boolean search(String word) {
        return root.searchII(word);
    }


    public static void main(String[] args) {
        WordDictionary wordDictionary = new WordDictionary();

        wordDictionary.addWord("a");

        wordDictionary.search("aa");

    }
}
