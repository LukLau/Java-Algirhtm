package org.dora.algorithm.geeksforgeek;

/**
 * @author dora
 * @date 2019/11/20
 */
public class SerialQuestion {


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
}
