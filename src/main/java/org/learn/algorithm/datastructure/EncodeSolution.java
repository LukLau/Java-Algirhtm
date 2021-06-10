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
        if (strs == null || strs.isEmpty()) {
            return "";
        }
        StringBuilder builder = new StringBuilder();
        int size = strs.size();
        for (int i = 0; i < size; i++) {
            String current = strs.get(i);
            builder.append(current);
            if (i != size - 1) {
                builder.append(size).append(";");
            }
        }
        return builder.toString();
        // write your code here
    }

    /**
     * @param str: A string
     * @return: dcodes a single string to a list of strings
     */
    public List<String> decode(String str) {
        if (str == null || str.isEmpty()) {
            return new ArrayList<>();
        }
        String[] words = str.split(";");
        List<String> result = new ArrayList<>();
        String endWord = String.valueOf(words.length);

        for (int i = 0; i < words.length; i++) {

            String word = words[i];
            if (i != words.length - 1) {
                int lastIndexOf = word.lastIndexOf(endWord);
                word = word.substring(0, lastIndexOf);
            }
            result.add(word);
        }

        return result;
        // write your code here
    }

    public static void main(String[] args) {
        EncodeSolution solution = new EncodeSolution();
        List<String> strs = new ArrayList<>();
        strs.add("lint");
        strs.add("code");
        strs.add("love");
        strs.add("your");
        String encode = solution.encode(strs);
        System.out.println(encode);
        List<String> decode = solution.decode("lint4;code4;love4;your");
        System.out.println(decode);
    }
}
