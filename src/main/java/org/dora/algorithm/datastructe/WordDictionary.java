package org.dora.algorithm.datastructe;

/**
 * mail lu.liu2@cariad-technology.cn
 * date 2024年05月18日
 * @author lu.liu2
 */
public class WordDictionary {

    public static void main(String[] args) {
        WordDictionary wordDictionary = new WordDictionary();
        wordDictionary.addWord("bad");
        wordDictionary.search(".ad");
    }

    private final PrefixTrie prefixTrie;

    public WordDictionary() {
        prefixTrie = new PrefixTrie();

    }

    public void addWord(String word) {
        prefixTrie.insert(word);

    }

    public boolean search(String word) {
        return prefixTrie.searchv2(word);
    }

    /**
     * mail lu.liu2@cariad-technology.cn
     * date 2024年05月18日
     * @author lu.liu2
     */
    public static class PrefixTrie {
        private final TrieNode root;

        public PrefixTrie() {
            root = new TrieNode();
        }

        public void insert(String word) {
            if (word == null || word.isEmpty()) {
                return;
            }
            TrieNode p = root;
            char[] words = word.toCharArray();

            for (char c : words) {
                if (p.sonTrees[c - 'a'] == null) {
                    p.sonTrees[c - 'a'] = new TrieNode();
                }
                p = p.sonTrees[c - 'a'];
            }
            p.word = word;
        }

        public boolean search(String word) {
            if (word == null || word.isEmpty()) {
                return true;
            }
            TrieNode p = root;
            char[] words = word.toCharArray();
            for (char c : words) {
                if (p.sonTrees[c - 'a'] == null) {
                    return false;
                }
                p = p.sonTrees[c - 'a'];
            }
            return word.equals(p.word);
        }


        public boolean startsWith(String prefix) {
            if (prefix == null || prefix.isEmpty()) {
                return true;
            }
            TrieNode p = root;
            char[] words = prefix.toCharArray();
            for (char c : words) {
                if (p.sonTrees[c - 'a'] == null) {
                    return false;
                }
                p = p.sonTrees[c - 'a'];
            }
            return true;
        }

        public boolean searchv2(String word) {
            if (word == null || word.isEmpty()) {
                return true;
            }
            char[] words = word.toCharArray();
            return internalSearch(root, 0, words);
        }

        private boolean internalSearch(TrieNode root, int index, char[] words) {
            if (index >= words.length) {
                return root.word != null;
            }
            char word = words[index];
            if (word != '.') {
                if (root.sonTrees[word - 'a'] == null) {
                    return false;
                }
                return internalSearch(root.sonTrees[word - 'a'], index + 1, words);
            }
            for (TrieNode sonTree : root.sonTrees) {
                if (sonTree != null && internalSearch(sonTree, index + 1, words)) {
                    return true;
                }
            }
            return false;
        }

        public static class TrieNode {
            public final TrieNode[] sonTrees;
            public String word;

            public TrieNode() {
                sonTrees = new TrieNode[26];
            }
        }

    }
}
