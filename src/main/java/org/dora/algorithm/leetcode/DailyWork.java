package org.dora.algorithm.leetcode;

/**
 * date 2024年04月25日
 */
public class DailyWork {

    public static void main(String[] args) {
        DailyWork dailyWork = new DailyWork();

        dailyWork.distanceTraveled(14, 10);
    }

    public int distanceTraveled(int mainTank, int additionalTank) {
        int count = 0;

        while (mainTank >= 5 && additionalTank > 0) {
            count += 50;

            mainTank -= 5;

            mainTank = Math.max(mainTank, 0);

            mainTank += 1;
            additionalTank--;
        }
        return count + 10 * mainTank;

//        int ans = 0;
//        while (mainTank >= 5) {
//            mainTank -= 5;
//            ans += 50;
//            if (additionalTank > 0) {
//                additionalTank--;
//                mainTank++;
//            }
//        }
//        return ans + mainTank * 10;

    }

}
