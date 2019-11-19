# 算法
    leetCode 以及 剑指 offer 算法相关题目
    
-----------

## 超时问题
> 超时问题往往代表代码正常    
由于存在重复计算问题导致超时、    
故需考虑记录已经计算的值或者状态、减少次数

> [除法问题](https://leetcode.com/problems/divide-two-integers/)
 
## 深度、广度优先问题
> 深度遍历靠栈来实现

> 宽度遍历靠队列实现

> 深度、广度遍历核心还是回溯法、回溯法需要注意更改回溯状态

> 遍历得考虑好边界退出条件

> [课程调度]()

> [八皇后问题](https://leetcode.com/problems/n-queens/)



## Dp问题
>**动态规划解题关键**

> 第一个要点需要考虑 子问题方程式

> 第二个要点需要考虑方程式连续问题。由于子问题决定下一个问题的最优解。故动态规划方程式必须连贯起来

> [魔法匹配问题](https://www.cnblogs.com/grandyang/p/4401196.html
)

> [正则表达式匹配](https://leetcode.com/problems/regular-expression-matching/discuss/5651/Easy-DP-Java-Solution-with-detailed-Explanation)

> [KMP算法](https://leetcode.com/problems/implement-strstr/discuss/12956/C%2B%2B-Brute-Force-and-KMP)

> [获取八皇后个数](https://leetcode.com/problems/n-queens-ii/)


## 贪心问题


## 分治问题

## 位操问题
    
* **两个不同数^ ==相当于 无进位加法**
* **两个不同数& 相当于 判断是否有进位**
* **& (偶数- 1) 相当于 取模操作**
* **n & 1 相当于 对2取模操作 相当于每次获取二进制最末尾一位数字值**
* **n & 1 并且 n 无符号右移相当于二进制数据反转**
> [二进制数据反转并且满足32数](https://leetcode.com/problems/reverse-bits/)



## 滑动窗口问题问题

* **考虑使用双指针**
* **考虑使用队列**
* **[滑动窗口模板](https://leetcode.com/problems/minimum-window-substring/discuss/26808/Here-is-a-10-line-template-that-can-solve-most-'substring'-problems)**

> 滑动窗口关键值

> **窗口的大小 当窗口太小时 窗口往右扩展。窗口太大时缩小窗口**

> **窗口左右边界问题。确定如何移除窗口条件**

> **窗口需要进行初始化。慎重考虑初始化条件**

> **可以考虑使用list存储窗口的值**

> 滑动窗口经典题目  

* [滑动窗口最大值](https://www.nowcoder.com/practice/1624bc35a45c42c0bc17d17fa0cba788?tpId=13&tqId=11217&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking&tPage=4) 使用队列    

* [滑动窗口最多两个无差别子串](http://www.cnblogs.com/grandyang/p/5185561.html)

* [窗口最小数量](https://leetcode.com/problems/minimum-size-subarray-sum/)


# 二分搜索
> trick 数组移动
> 从左往右 (left + right) / 2 + 1
> 从右往左 (left + right) / 2 -1

> 边界值left <= right 或者 left < right 取决于代码思路

* [34. Find First and Last Position of Element in Sorted Array](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/)





----

## 比较难题目
> [两个数取中位数](https://leetcode.com/problems/median-of-two-sorted-arrays/)


-----
### 常用数学理论以及算法

> 空集是任何一个集合的子集

> [约瑟夫环](https://www.nowcoder.com/practice/f78a359491e64a50bce2d89cff857eb6?tpId=13&tqId=11199&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking&tPage=3)

> [格雷码以及二进制编码成格雷码](https://baike.baidu.com/item/%E6%A0%BC%E9%9B%B7%E7%A0%81/6510858?fr=aladdin)

> [求解质数的个数](https://leetcode.com/submissions/detail/121785675/)

> [约瑟夫环三种解法](https://blog.csdn.net/weixin_38214171/article/details/80352921)

> [计算素数个数](https://leetcode.com/problems/count-primes/)

> [字典树](https://leetcode.com/submissions/detail/226455100/)

> [摩尔投票法](https://www.jianshu.com/p/c19bb428f57a) **关键在于找出候选者**

> [位运算](https://leetcode.com/problems/single-number-ii/) 需要使用两位

> [逆波兰数](https://leetcode.com/problems/reverse-words-in-a-string/)

----
## 位运算
------

> [二进制反转](https://leetcode.com/problems/reverse-bits/)

思路: 进行位运算

---
# 遍历问题

> [树层次遍历O(1)空间](https://leetcode.com/problems/populating-next-right-pointers-in-each-node/)

> [树层次遍历II O(1)空间](https://leetcode.com/problems/populating-next-right-pointers-in-each-node-ii/)