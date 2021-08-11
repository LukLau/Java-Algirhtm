package org.learn.algorithm.datastructure;

import java.util.ArrayList;
import java.util.List;

/**
 * 字符串编码问题
 *
 * @author luk
 * @date 2021/4/20
 */
public class EncodeSolution {

    /**
     * @param strs: a list of strings
     * @return: encodes a list of strings to a single string.
     */
    public String encode(List<String> strs) {
        // write your code here
        if (strs == null || strs.isEmpty()) {
            return "";
        }
        StringBuilder builder = new StringBuilder();
        int len = strs.size();
        for (int i = 0; i < len; i++) {
            String current = strs.get(i);
            builder.append(current);
            if (i != len - 1) {
                builder.append(len).append(",");
            }
        }
        return builder.toString();
    }

    /**
     * @param str: A string
     * @return: dcodes a single string to a list of strings
     */
    public List<String> decode(String str) {
        if (str == null || str.isEmpty()) {
            return new ArrayList<>();
        }
//        String[] words = str.split(";");
//        List<String> result = new ArrayList<>();
//        String endWord = String.valueOf(words.length);
//
//        for (int i = 0; i < words.length; i++) {
//
//            String word = words[i];
//            if (i != words.length - 1) {
//                int lastIndexOf = word.lastIndexOf(endWord);
//                word = word.substring(0, lastIndexOf);
//            }
//            result.add(word);
//        }
//
//        return result;
        String[] words = str.split(",");
        List<String> result = new ArrayList<>();
        String len = String.valueOf(words.length);
        for (int i = 0; i < words.length; i++) {
            String word = words[i];
            if (i != words.length - 1) {
                int lastIndex = word.lastIndexOf(len);
                word = word.substring(0, lastIndex);
            }
            result.add(word);
        }
        // write your code here
        return result;
    }

}
