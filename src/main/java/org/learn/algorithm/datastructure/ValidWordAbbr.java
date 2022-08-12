package org.learn.algorithm.datastructure;

import java.util.HashMap;
import java.util.Map;

/**
 * 288
 * Unique Word Abbreviation
 *
 * @author luk
 * @date 2021/4/22
 */
public class ValidWordAbbr {

    private final Map<String, Integer> abbrMap = new HashMap<>();

    private final Map<String, Integer> wordMap = new HashMap<>();

    /*
     * @param dictionary: a list of words
     */
    public ValidWordAbbr(String[] dictionary) {
        // do intialization if necessary
        for (String word : dictionary) {

            int wordCount = wordMap.getOrDefault(word, 0);

            wordMap.put(word, wordCount + 1);

            String abbrWord = getAbbrWord(word);

//            if (!word.equals(abbrWord)) {

                int count = abbrMap.getOrDefault(abbrWord, 0);

                abbrMap.put(abbrWord, count + 1);
//            }


        }
    }

    /*
     * @param word: a string
     * @return: true if its abbreviation is unique or false
     */
    public boolean isUnique(String word) {
        String abbr = getAbbrWord(word);
        // write your code here"
        return wordMap.getOrDefault(word, 0).equals(abbrMap.getOrDefault(abbr, 0));
    }


    private String getAbbrWord(String word) {
        int len = word.length();
        if (len <= 2) {
            return word;
        }
        return String.valueOf(word.charAt(0)) +
                (len - 2) +
                word.charAt(len - 1);
    }


    public static void main(String[] args) {
        String[] word = new String[]{"deer", "door", "cake", "card"};
        ValidWordAbbr validWordAbbr = new ValidWordAbbr(word);

//        System.out.println(validWordAbbr.isUnique("dear"));
//        System.out.println(validWordAbbr.isUnique("cart"));
        System.out.println(validWordAbbr.isUnique("cane"));
        System.out.println(validWordAbbr.isUnique("make"));


    }


}
