package org.learn.algorithm.leetcode;

import javax.sound.midi.Track;
import java.util.ArrayList;
import java.util.List;

public class VipRecursive {

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
        internalAddOperators(result, num, "", "", 0, 0, target);
        return result;
    }

    private void internalAddOperators(List<String> result, String num, String s, String sign, int val, int previous, int target) {
        if (val == target) {
            result.add(s);
            return;
        }
        for (int i = 0; i < s.length(); i++) {
            String substring = s.substring(0, i + 1);
            int parseValue = Integer.parseInt(substring);
            String remain = s.substring(i + 1);


            if (!sign.isEmpty()) {
                num = num + sign + substring;
            } else {
                num = substring;
            }
            if (sign.equals("+")) {
                val += parseValue;
            } else if (sign.equals("-")) {
                val -= parseValue;
            }  else if (sign.equals("*")) {
                val -=  previous + previous * parseValue;
            } else if (sign.equals("/")) {

            }
            if (sign.isEmpty()) {
                internalAddOperators(result, substring, remain, "+", parseValue, parseValue, target);
                internalAddOperators(result, substring, remain, "-", parseValue, parseValue, target);
                internalAddOperators(result, substring, remain, "*", parseValue, parseValue, target);
                internalAddOperators(result, substring, remain, "/", parseValue, parseValue, target);
            } else {
                if (sign.equals("*")) {
//                    internalAddOperators(result, num + sign + substring, , , , , );
                }

            }
        }

    }

}
