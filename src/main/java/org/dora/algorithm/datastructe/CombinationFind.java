package org.dora.algorithm.datastructe;

/**
 * mail lu.liu2@cariad-technology.cn
 * date 2024年08月01日
 * @author lu.liu2
 */
public class CombinationFind {

    private int[] fathers = null;

    public int findFather(int number) {
        if (fathers[number] == number) {
            return number;
        }
        return fathers[number] = findFather(fathers[number]);
    }


    public void join(int oldNode, int newNode) {
        int oldFather = findFather(oldNode);
        int newFather = findFather(newNode);

        if (oldFather == newFather) {
            return;
        }
        if (oldFather < newFather) {
            fathers[newFather] = oldFather;
        } else {
            fathers[oldFather] = newFather;
        }
    }

    public CombinationFind(int n) {
        fathers = new int[n];
        for (int i = 0; i < fathers.length; i++) {
            fathers[i] = i;
        }
    }

    public int[] getFathers() {
        return fathers;
    }
}
