package org.dora.algorithm.swordoffer;

/**
 * @author liulu
 * @date 2019/04/24
 */
public class SwordToOffer {
    /**
     * 二维数组的查找
     *
     * @param target
     * @param array
     * @return
     */
    public boolean Find(int target, int[][] array) {
        if (array == null || array.length == 0) {
            return false;
        }
        int row = array.length;
        int column = array[0].length;
        int i = row - 1;
        int j = 0;
        while (i >= 0 && j < column) {
            if (array[i][j] == target) {
                return true;
            } else if (array[i][j] < target) {
                j++;
            } else {
                i--;
            }
        }
        return false;
    }

    /**
     * 替换空格
     *
     * @param str
     * @return
     */
    public String replaceSpace(StringBuffer str) {
        if (str == null || str.length() == 0) {
            return "";
        }
        String value = str.toString();
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i <value.length(); i++) {
            if (value.charAt(i) == ' ') {
                sb.append("%20");
            } else {
                sb.append(value.charAt(i));
            }
        }
        return sb.toString();
    }
}
