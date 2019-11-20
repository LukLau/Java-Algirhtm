package org.dora.algorithm.geeksforgeek;

/**
 * @author dora
 * @date 2019/11/20
 */
public class SerialQuestion {


    //    --- 卖股票 ----- //

    /**
     * 121. Best Time to Buy and Sell Stock
     *
     * @param prices
     * @return
     */
    public int maxProfit(int[] prices) {
        if (prices == null || prices.length == 0) {
            return 0;
        }
        int minPrice = prices[0];
        int result = 0;
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > minPrice) {
                result = Math.max(result, prices[i] - minPrice);
            } else {
                minPrice = prices[i];
            }
        }
        return result;
    }

    /**
     * 122. Best Time to Buy and Sell Stock II
     *
     * @param prices
     * @return
     */
    public int maxProfitII(int[] prices) {
        if (prices == null || prices.length == 0) {
            return 0;
        }
        int result = 0;

        int minPrice = prices[0];
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > minPrice) {
                result += prices[i] - minPrice;
            }
            minPrice = prices[i];
        }
        return result;
    }

    /**
     * todo 股票交易两次
     * 123. Best Time to Buy and Sell Stock III
     *
     * @param prices
     * @return
     */
    public int maxProfitIII(int[] prices) {
        if (prices == null || prices.length == 0) {
            return 0;
        }
        int n = prices.length;

        int[] leftProfit = new int[n];

        int minLeftPrice = prices[0];

        int leftResult = 0;

        for (int i = 1; i < n; i++) {
            if (prices[i] < minLeftPrice) {
                minLeftPrice = prices[i];
            }

            leftResult = Math.max(leftResult, prices[i] - minLeftPrice);

            leftProfit[i] = leftResult;
        }
        int[] rightProfit = new int[n + 1];

        int maxRightPrice = prices[n - 1];

        int rightResult = 0;

        for (int i = n - 2; i >= 1; i--) {
            if (maxRightPrice < prices[i]) {

                maxRightPrice = prices[i];
            }

            rightResult = Math.max(rightResult, maxRightPrice - prices[i]);

            rightProfit[i] = rightResult;

        }

        int result = 0;

        for (int i = 0; i < n; i++) {
            result = Math.max(result, leftProfit[i] + rightProfit[i + 1]);
        }
        return result;


    }
}
