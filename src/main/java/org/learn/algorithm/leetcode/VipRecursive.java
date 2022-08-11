package org.learn.algorithm.leetcode;

import javax.sound.midi.Track;
import java.util.ArrayList;
import java.util.List;

public class VipRecursive {

    public static void main(String[] args) {
        VipRecursive vipRecursive = new VipRecursive();
        System.out.println(vipRecursive.addOperators("123", 6));

    }

    /**
     * @param num:    a string contains only digits 0-9
     * @param target: An integer
     * @return: return all possibilities
     * we will sort your return value in output
     */
    public List<String> addOperators(String num, int target) {
        // write your code here
        if (num == null || num.isEmpty()) {
            return new ArrayList<>();
        }
        List<String> result = new ArrayList<>();
        internalAddOperators(result, "", num, "", 0, 0, target);
        return result;
    }

    private void internalAddOperators(List<String> result, String num, String s, String sign, int val, int previous, int target) {
        if (s.isEmpty()) {
            return;
        }
        for (int i = 0; i < s.length(); i++) {
            String substring = s.substring(0, i + 1);
            int parseValue = Integer.parseInt(substring);
            String remain = s.substring(i + 1);


            if (!sign.isEmpty()) {
//                num = num + sign + substring;
//            } else {
                val = Integer.parseInt(substring);
//                num = substring;
            }
            if (sign.equals("+")) {
                val += parseValue;
            } else if (sign.equals("-")) {
                val -= parseValue;
            } else if (sign.equals("*")) {
                val = val - previous + previous * parseValue;
            } else if (sign.equals("/")) {
                val = val - previous + previous / parseValue;
            }
            if (val == target) {
                result.add(num);
            }
            internalAddOperators(result, num + sign + substring, remain, "+", val, parseValue, target);
            internalAddOperators(result, num + sign + substring, remain, "-", val, parseValue, target);
            internalAddOperators(result, num + sign + substring, remain, "*", val, parseValue, target);
            internalAddOperators(result, num + sign + substring, remain, "/", val, parseValue, target);
        }

    }

}
