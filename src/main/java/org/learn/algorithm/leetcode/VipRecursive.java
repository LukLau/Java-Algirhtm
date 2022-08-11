package org.learn.algorithm.leetcode;

import javax.sound.midi.Track;
import java.security.cert.CertificateParsingException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class VipRecursive {

    public static void main(String[] args) {
        VipRecursive vipRecursive = new VipRecursive();
        System.out.println(vipRecursive.addOperators("101",
                45));

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
        Collections.sort(result);
        return result;
    }

    private void internalAddOperators(List<String> result, String num, String s, String sign, long val, long previous, long target) {
        if (s.isEmpty()) {
            return;
        }

        for (int i = 0; i < s.length(); i++) {
            String substring = s.substring(0, i + 1);
            if (substring.length() >= 2 && substring.charAt(0) == '0') {
                continue;
            }
            long parseValue = Long.parseLong(substring);

            String remain = s.substring(i + 1);

            long tmpValue;
            if (sign.equals("+")) {
                tmpValue = val + parseValue;
            } else if (sign.equals("-")) {
                tmpValue = val - parseValue;
                parseValue = -parseValue;
            } else if (sign.equals("*")) {
                tmpValue = val - previous + previous * parseValue;
            } else {
                tmpValue = parseValue;
            }

            String tmp = num + sign + substring;

            if (tmpValue == target && remain.isEmpty()) {
                result.add(tmp);
            }
            internalAddOperators(result, tmp, remain, "*", tmpValue, sign.isEmpty() ? tmpValue : previous * parseValue, target);
            internalAddOperators(result, tmp, remain, "+", tmpValue, parseValue, target);
            internalAddOperators(result, tmp, remain, "-", tmpValue, parseValue, target);
        }

    }

}
