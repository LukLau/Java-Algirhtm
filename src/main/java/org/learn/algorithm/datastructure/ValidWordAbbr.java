package org.learn.algorithm.datastructure;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * 288
 * Unique Word Abbreviation
 *
 * @author luk
 * @date 2021/4/22
 */
public class ValidWordAbbr {

    private final Map<String, Integer> words = new HashMap<>();

    private final Map<String, Integer> abbrWords = new HashMap<>();


    /*
     * @param dictionary: a list of words
     */
    public ValidWordAbbr(String[] dictionary) {
        // do intialization if necessary
        for (String word : dictionary) {
            String abbrWord = getAbbrWord(word);

            Integer originalCount = words.getOrDefault(word, 0);

            words.put(word, originalCount + 1);

            Integer abbrWordCount = abbrWords.getOrDefault(abbrWord, 0);

            abbrWords.put(abbrWord, abbrWordCount + 1);
        }
    }

    /*
     * @param word: a string
     * @return: true if its abbreviation is unique or false
     */
    public boolean isUnique(String word) {
        // write your code here
        Integer originalCount = words.get(word);


        String abbrWord = getAbbrWord(word);

        Integer abbrWordCount = abbrWords.get(abbrWord);

        return Objects.equals(originalCount, abbrWordCount);

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


    /**
     * https://www.lintcode.com/problem/637
     *
     * @param word: a non-empty string
     * @param abbr: an abbreviation
     * @return: true if string matches with the given abbr or false
     */
    public boolean validWordAbbreviation(String word, String abbr) {
        // write your code here
        if (word == null || abbr == null) {
            return false;
        }
        int len = word.length();

        for (int i = 1; i <= len; i++) {

        }
        return false;

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
