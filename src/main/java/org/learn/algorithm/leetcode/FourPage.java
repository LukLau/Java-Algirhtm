package org.learn.algorithm.leetcode;

import java.util.*;

/**
 * leetcode the four page
 *
 * @author luk
 * @date 2021/4/26
 */
public class FourPage {

    public static void main(String[] args) {
        FourPage fourPage = new FourPage();
//        fourPage.generateAbbreviations("word");
//        String pattern = "abba";
//        String s = "dog cat cat dog";
//        fourPage.wordPattern(pattern, s);

        String pattern = "ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccdd";
        String word = "s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s s t t";
//        fourPage.wordPattern(pattern, word);

//        pattern = "abab";
//        word = "redblueredblue";

        pattern = "bdpbibletwuwbvh";
        word = "aaaaaaaaaaaaaaa";
        fourPage.wordPatternMatch(pattern, word);
    }


    /**
     * 301. Remove Invalid Parentheses
     *
     * @param s
     * @return
     */
    public List<String> removeInvalidParentheses(String s) {
        if (s == null) {
            return new ArrayList<>();
        }
        List<String> result = new ArrayList<>();
        LinkedList<String> linkedList = new LinkedList<>();
        linkedList.offer(s);
        Set<String> seen = new HashSet<>();
        while (!linkedList.isEmpty()) {
            String poll = linkedList.poll();
            if (checkValid(poll) && !result.contains(poll)) {
                result.add(poll);
            }
            if (!result.isEmpty()) {
                continue;
            }
            int len = poll.length();
            for (int i = 0; i < len; i++) {
                char word = poll.charAt(i);
                if (word != '(' && word != ')') {
                    continue;
                }
                String tmp = poll.substring(0, i) + poll.substring(i + 1);
                if (!seen.contains(tmp)) {
                    linkedList.offer(tmp);
                    seen.add(tmp);
                }
            }
        }
        if (result.isEmpty()) {
            result.add("");
        }
        return result;
    }

    private boolean checkValid(String s) {
        int count = 0;
        char[] words = s.toCharArray();
        for (char word : words) {
            if (word != '(' && word != ')') {
                continue;
            }
            if (word == '(') {
                count++;
            }
            if (word == ')') {
                if (count == 0) {
                    return false;
                }
                count--;
            }
        }
        return count == 0;
    }


    public boolean isAdditiveNumber(String num) {
        if (num == null || num.isEmpty()) {
            return true;
        }
        int len = num.length();
        for (int i = 1; i < len; i++) {
            if (num.charAt(0) == '0' && i > 1) {
                return false;
            }
            for (int j = i + 1; len - j >= j - i && len - j >= i; j++) {
                if (num.charAt(j) == '0' && j - i > 2) {
                    break;
                }
                long num1 = Long.parseLong(num.substring(0, i));
                long num2 = Long.parseLong(num.substring(i, i + j));
                if (isAdditive(num.substring(i + j), num1, num2)) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean isAdditive(String substring, long num1, long num2) {
        if (substring.isEmpty()) {
            return true;
        }
        long sum = num1 + num2;
        String value = String.valueOf(sum);
        if (!substring.startsWith(value)) {
            return false;
        }
        return isAdditiveNumber(substring.substring(value.length()));
    }


    /**
     * 316. Remove Duplicate Letters
     *
     * @param s
     * @return
     */
    public String removeDuplicateLetters(String s) {
        if (s == null || s.isEmpty()) {
            return "";
        }
        return "";
    }

    /**
     * 318. Maximum Product of Word Lengths
     * todo
     *
     * @param words
     * @return
     */
    public int maxProduct(String[] words) {
        if (words == null || words.length == 0) {
            return 0;
        }
        int result = 0;
        for (String word : words) {
            for (String s : words) {
                if (s.equals(word)) {
                    continue;
                }
                boolean isPrefix = false;
                char[] tmp = s.toCharArray();
                for (char t : tmp) {
                    if (word.indexOf(t) != -1) {
                        isPrefix = true;
                        break;
                    }
                }
                if (!isPrefix) {
                    result = Math.max(result, word.length() * s.length());
                }
            }
        }
        return result;
    }


    /**
     * todo
     * 320
     * Generalized Abbreviation
     *
     * @param word: the given word
     * @return: the generalized abbreviations of a word
     */
    public List<String> generateAbbreviations(String word) {
        // Write your code here
        if (word == null || word.isEmpty()) {
            return new ArrayList<>();
        }
        List<String> result = new ArrayList<>();
        intervalGenerate(result, 0, 0, "", word);
        return result;
    }

    private void intervalGenerate(List<String> result, int count, int pos, String s, String word) {
        if (pos == word.length()) {
            if (count > 0) {
                s += count;
            }
            result.add(s);
            return;
        }
        intervalGenerate(result, count + 1, pos + 1, s, word);
        if (count > 0) {
            s = s + count + word.charAt(pos);
        } else {
            s = s + word.charAt(pos);
        }
        intervalGenerate(result, 0, pos + 1, s, word);
    }

    /**
     * 283. Move Zeroes
     *
     * @param nums
     */
    public void moveZeroes(int[] nums) {
        if (nums == null || nums.length == 0) {
            return;
        }
        int zeroIndex = 0;
//        int endIndex = nums.length - 1;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != 0) {
                swap(nums, zeroIndex++, i);
            }
        }
//        while (zeroIndex < endIndex) {
//            if (nums[endIndex] != 0) {
//                swap(nums, endIndex, zeroIndex++);
//            }
//            endIndex--;
//        }
    }

    private void swap(int[] nums, int start, int end) {
        int val = nums[start];
        nums[start] = nums[end];
        nums[end] = val;
    }


    /**
     * 290. Word Pattern
     *
     * @param pattern
     * @param s
     * @return
     */
    public boolean wordPattern(String pattern, String s) {
        if (pattern == null || s == null) {
            return false;
        }
        Map<Character, Integer> map1 = new HashMap<>();
        Map<String, Integer> map2 = new HashMap<>();
        String[] words = s.split(" ");

        int len = pattern.length();
        if (words.length != len) {
            return false;
        }
        for (int i = 0; i < len; i++) {
            char tmp = pattern.charAt(i);

            Integer firstIndex = map1.get(tmp);

            Integer secondIndex = map2.get(words[i]);

            if (Objects.equals(firstIndex, secondIndex)) {
                return false;
            }
            map1.put(tmp, i);
            map2.put(words[i], i);
        }
        return true;
    }


    /**
     * 291 Word Pattern II
     *
     * @param pattern: a string,denote pattern string
     * @param str:     a string, denote matching string
     * @return: a boolean
     */
    public boolean wordPatternMatch(String pattern, String str) {
        // write your code here
        if (pattern == null || str == null) {
            return false;
        }
        Map<Character, String> map = new HashMap<>();
        Set<String> visited = new HashSet<>();
        return internalWordPatternMatch(pattern, str, map, visited);
    }

    private boolean internalWordPatternMatch(String pattern, String str, Map<Character, String> map, Set<String> visited) {
        if (pattern.isEmpty()) {
            return str.isEmpty();
        }
        String prefix = map.getOrDefault(pattern.charAt(0), "");

        if (!prefix.isEmpty()) {
            if (!str.startsWith(prefix)) {
                return false;
            }
            return internalWordPatternMatch(pattern.substring(1), str.substring(prefix.length()), map, visited);
        }
        int len = str.length();

        for (int i = 1; i <= len; i++) {
            String substring = str.substring(0, i);

            if (visited.contains(substring)) {
                continue;
            }
            visited.add(substring);
            map.put(pattern.charAt(0), substring);

            String remain = str.substring(i);

            if (internalWordPatternMatch(pattern.substring(1), remain, map, visited)) {
                return true;
            }
            visited.remove(substring);
            map.remove(pattern.charAt(0));
        }
        return false;
    }

}
