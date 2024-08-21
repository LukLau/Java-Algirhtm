package org.dora.algorithm.leetcode;

import org.dora.algorithm.datastructe.ListNode;
import org.dora.algorithm.datastructe.Node;
import sun.util.resources.cldr.ka.LocaleNames_ka;

import java.util.*;

/**
 * date 2024年04月16日
 */
public class TopInterview {

    public int minSubArrayLen(int target, int[] nums) {
        int result = Integer.MAX_VALUE;
        int left = 0;

        int sum = 0;
        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];
            while (sum >= target) {
                result = Math.min(result, i - left + 1);

                sum -= nums[left++];
            }
        }
        return result;
    }


    public int lengthOfLongestSubstring(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        }
        int left = 0;
        int result = 0;
        Map<Character, Integer> map = new HashMap<>();
        char[] words = s.toCharArray();
        for (int i = 0; i < words.length; i++) {
            if (map.containsKey(words[i])) {
                left = Math.max(left, map.get(words[i]) + 1);
            }
            result = Math.max(result, i - left + 1);

            map.put(words[i], i);
        }
        return result;
    }


    public String minWindow(String s, String t) {
        int[] hash = new int[256];

        for (char tmp : t.toCharArray()) {
            hash[tmp]++;
        }
        int result = Integer.MAX_VALUE;
        int begin = 0;
        int end = 0;
        int head = 0;
        int count = t.length();
        while (end < s.length()) {
            if (hash[s.charAt(end++)]-- > 0) {
                count--;
            }
            while (count == 0) {
                if (end - begin < result) {
                    result = end - begin;
                    head = begin;
                }
                if (hash[s.charAt(begin++)]++ == 0) {
                    count++;
                }
            }
        }
        if (result == Integer.MAX_VALUE) {
            return "";
        }
        return s.substring(head, head + result);

    }

    public void gameOfLife(int[][] board) {
        if (board == null || board.length == 0) {
            return;
        }
        int row = board.length;
        int column = board[0].length;
        int[][] matrix = new int[][]{{-1, 0}, {1, 0}, {0, 1}, {0, -1}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                int liveCount = 0;
                for (int[] iterator : matrix) {
                    int x = i + iterator[0];
                    int y = j + iterator[1];
                    if (x < 0 || x >= row || y < 0 || y >= column) {
                        continue;
                    }
//                    if (Math.abs(board[x][y]) >= 1) {
//                        liveCount++;
//                    }

                    if (Math.abs(board[x][y]) == 1) {
                        liveCount++;
                    }
                }
                if (board[i][j] == 1 && (liveCount < 2 || liveCount > 3)) {
                    board[i][j] = -1;
                }
                if (board[i][j] == 0 && liveCount == 3) {
                    board[i][j] = -2;
                }
            }
        }
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (board[i][j] == -1) {
                    board[i][j] = 0;
                } else if (board[i][j] == -2) {
                    board[i][j] = 1;
                }
            }
        }
    }

    public boolean canConstruct(String ransomNote, String magazine) {
        Map<Character, Integer> map = new HashMap<>();

        for (char c : ransomNote.toCharArray()) {
            Integer count = map.getOrDefault(c, 0);
            count++;
            map.put(c, count);
        }
        for (char c : magazine.toCharArray()) {
            if (map.containsKey(c)) {
                Integer count = map.get(c);
                count--;

                if (count == 0) {
                    map.remove(c);
                } else {
                    map.put(c, count);
                }
            }
        }
        return map.isEmpty();

    }

    public boolean isIsomorphic(String s, String t) {
        if (s.length() != t.length()) {
            return false;
        }
        Map<Character, Integer> map1 = new HashMap<>();
        Map<Character, Integer> map2 = new HashMap<>();

        for (int i = 0; i < s.toCharArray().length; i++) {

            Integer left1 = map1.getOrDefault(s.charAt(i), i);
            Integer left2 = map2.getOrDefault(t.charAt(i), i);

            if (!Objects.equals(left1, left2)) {
                return false;
            }
            map1.put(s.charAt(i), i);
            map2.put(t.charAt(i), i);
        }
        return true;
    }

    public boolean wordPattern(String pattern, String s) {
        String[] words = s.split(" ");
        if (pattern.length() != words.length) {
            return false;
        }

        Map<Character, Integer> map1 = new HashMap<>();
        Map<String, Integer> map2 = new HashMap<>();

        for (int i = 0; i < pattern.toCharArray().length; i++) {
            Integer left1 = map1.getOrDefault(pattern.charAt(i), i);
            Integer left2 = map2.getOrDefault(words[i], i);
            ;
            if (!Objects.equals(left1, left2)) {
                return false;
            }
            map1.put(pattern.charAt(i), i);
            map2.put(words[i], i);
        }
        return true;

    }

    public boolean isAnagram(String s, String t) {
        if (s.length() != t.length()) {
            return false;
        }

        int[] hash = new int[256];
        int m = s.length();
        for (int i = 0; i < m; i++) {
            hash[s.charAt(i)]++;
            hash[t.charAt(i)]--;
        }
        for (int count : hash) {
            if (count != 0) {
                return false;
            }
        }
        return true;

    }

    public boolean isHappy(int n) {
        if (n < 10) {
            return false;
        }
        Set<Integer> visited = new HashSet<>();

        while (n != 0) {
            if (visited.contains(n)) {
                return false;
            }
            visited.add(n);

            int tmp = n;
            int number = 0;
            while (tmp != 0) {
                int remain = tmp % 10;

                number += remain * remain;

                tmp /= 10;
            }
            if (number == 1) {
                return true;
            }
            n = number;
        }
        return false;
    }

    public int evalRPN(String[] tokens) {
        if (tokens == null || tokens.length == 0) {
            return 0;
        }
        Stack<Integer> stack = new Stack<>();
        for (String token : tokens) {
            if (Objects.equals(token, "+")) {

                Integer firstItem = stack.pop();
                Integer secondItem = stack.pop();

                stack.push(firstItem + secondItem);
            } else if (Objects.equals(token, "-")) {
                Integer firstItem = stack.pop();
                Integer secondItem = stack.pop();
                stack.push(secondItem - firstItem);
            } else if (Objects.equals(token, "*")) {
                Integer firstItem = stack.pop();
                Integer secondItem = stack.pop();
                stack.push(secondItem * firstItem);
            } else if (Objects.equals(token, "/")) {
                Integer firstItem = stack.pop();
                Integer secondItem = stack.pop();
                stack.push(secondItem / firstItem);

            } else {
                stack.push(Integer.parseInt(token));
            }
        }
        return stack.pop();
    }

    public int minimumTotal(List<List<Integer>> triangle) {
        if (triangle == null || triangle.isEmpty()) {
            return 0;
        }
        int size = triangle.size();
        List<Integer> lastRow = triangle.get(size - 1);

        for (int i = size - 2; i >= 0; i--) {
            for (int j = 0; j < triangle.get(i).size(); j++) {
                int minValue = Math.min(triangle.get(i + 1).get(j), triangle.get(i + 1).get(j + 1)) + triangle.get(i).get(j);

                triangle.get(i).set(j, minValue);

            }
        }
        return triangle.get(0).get(0);
    }

    public String longestPalindrome(String s) {
        if (s == null || s.isEmpty()) {
            return "";
        }
        int len = s.length();
        boolean[][] dp = new boolean[len][len];

        int result = Integer.MIN_VALUE;
        int begin = 0;
        for (int i = 0; i < len; i++) {
            for (int j = 0; j <= i; j++) {
                if (s.charAt(j) == s.charAt(i) && (i - j <= 2 || dp[j + 1][i - 1])) {
                    dp[j][i] = true;
                }
                if (dp[j][i] && (i - j + 1 > result)) {
                    result = i - j + 1;
                    begin = j;
                }
            }
        }
        if (result == Integer.MIN_VALUE) {
            return "";
        }
        return s.substring(begin, begin + result);
    }


    public boolean isInterleave(String s1, String s2, String s3) {
        if (s1.isEmpty() && s2.isEmpty()) {
            return s3.isEmpty();
        }

        int m = s1.length();
        int n = s2.length();
        int len = s3.length();
        if (m + n != len) {
            return false;
        }
        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[0][0] = true;
        for (int i = 1; i <= m; i++) {
            dp[i][0] = s1.charAt(i - 1) == s3.charAt(i - 1) && dp[i - 1][0];
        }
        for (int j = 1; j <= n; j++) {
            dp[0][j] = s2.charAt(j - 1) == s3.charAt(j - 1) && dp[0][j - 1];
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                dp[i][j] = (s1.charAt(i - 1) == s3.charAt(i + j - 1) && dp[i - 1][j]) || (s2.charAt(j - 1) == s3.charAt(i + j - 1) && dp[i][j - 1]);
            }
        }
        return dp[m][n];
    }

    public int maxProfit(int[] prices) {
        if (prices == null || prices.length == 0) {
            return 0;
        }
        int[] leftProfit = new int[prices.length];
        int leftCost = prices[0];
        int leftMaxProfit = 0;
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > leftCost) {
                leftMaxProfit = Math.max(leftMaxProfit, prices[i] - leftCost);
            } else {
                leftCost = prices[i];
            }
            leftProfit[i] = leftMaxProfit;
        }
        int[] rightProfit = new int[prices.length + 1];
        int rightSell = prices[prices.length - 1];
        int rightMaxProfit = 0;

        for (int i = prices.length - 2; i >= 0; i--) {
            if (prices[i] < rightSell) {
                rightMaxProfit = Math.max(rightMaxProfit, rightSell - prices[i]);
            } else {
                rightSell = prices[i];
            }
            rightProfit[i] = rightMaxProfit;
        }
        int result = 0;
        for (int i = 1; i < prices.length; i++) {
            result = Math.max(result, leftProfit[i] + rightProfit[i + 1]);
        }
        return result;
    }


    public int maxProfit(int k, int[] prices) {
        if (prices == null || prices.length == 0) {
            return 0;
        }
        int[][] dp = new int[k + 1][prices.length];
        for (int i = 1; i <= k; i++) {
            int cost = -prices[0];
            for (int j = 1; j < prices.length; j++) {
                dp[i][j] = Math.max(dp[i][j - 1], cost + prices[j]);

                cost = Math.max(cost, dp[i - 1][j - 1] - prices[j]);
            }
        }
        return dp[k][prices.length - 1];
    }

    public int maximalSquare(char[][] matrix) {
        if (matrix == null || matrix.length == 0) {
            return 0;
        }
        int row = matrix.length;
        int column = matrix[0].length;

        int[][] dp = new int[row][column];
        int result = 0;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (matrix[i][j] == '1') {
                    if (i == 0 || j == 0) {
                        dp[i][j] = 1;
                    } else {
                        dp[i][j] = Math.min(Math.min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]) + 1;
                    }
                    result = Math.max(result, dp[i][j] * dp[i][j]);
                }
            }
        }
        return result;
    }

    public int climbStairs(int n) {
        if (n == 1) {
            return 1;
        }
        if (n == 2) {
            return 2;
        }
        int[] hash = new int[n + 1];
        hash[1] = 1;
        hash[2] = 2;
        return internalClimb(hash, n);
    }

    private int internalClimb(int[] hash, int n) {
        if (hash[n] != 0) {
            return hash[n];
        }
        int result = internalClimb(hash, n - 1) + internalClimb(hash, n - 2);
        hash[n] = result;
        return result;
    }


    public boolean wordBreak(String s, List<String> wordDict) {
        if (s == null || s.isEmpty()) {
            return true;
        }
        Map<String, Boolean> map = new HashMap<>();
        return internalWordBreak(map, s, wordDict);
    }

    private boolean internalWordBreak(Map<String, Boolean> map, String s, List<String> wordDict) {
        if (s == null || s.isEmpty()) {
            return true;
        }
        if (map.containsKey(s)) {
            return map.get(s);
        }
        for (String string : wordDict) {
            if (s.startsWith(string) && internalWordBreak(map, s.substring(string.length()), wordDict)) {
                return true;

            }
        }
        map.put(s, false);
        return false;
    }

    public boolean wordBreakii(String s, List<String> wordDict) {
        if (s == null || s.isEmpty()) {
            return true;
        }
        boolean[] dp = new boolean[s.length() + 1];
        dp[0] = true;
        for (int i = 1; i <= s.length(); i++) {
            for (int j = 0; j < i; j++) {
                if (dp[j] && wordDict.contains(s.substring(j, i))) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[s.length()];
    }


    public int maxPoints(int[][] points) {
        if (points == null || points.length == 0) {
            return 0;
        }
        int result = 0;
        for (int i = 0; i < points.length; i++) {
            int overlap = 0;
            Map<Integer, Map<Integer, Integer>> map = new HashMap<>();
            int number = 0;
            for (int j = i + 1; j < points.length; j++) {
                int x = points[j][0] - points[i][0];
                int y = points[j][1] - points[i][1];


                if (x == 0 && y == 0) {
                    overlap++;
                    continue;
                }
                int gcd = gcd(x, y);

                x /= gcd;
                y /= gcd;

                if (!map.containsKey(x)) {
                    Map<Integer, Integer> tmp = new HashMap<>();
                    tmp.put(y, 1);
                    map.put(x, tmp);
                } else {
                    Map<Integer, Integer> tmp = map.get(x);
                    Integer cnt = tmp.getOrDefault(y, 0);
                    cnt++;
                    tmp.put(y, cnt);
                }
                number = Math.max(number, map.get(x).get(y));
            }
            result = Math.max(result, overlap + number + 1);
        }
        return result;
    }

    private int gcd(int x, int y) {
        if (y == 0) {
            return x;
        }
        return gcd(y, x % y);
    }

    public int trailingZeroes(int n) {
        int result = 0;
        while (n / 5 != 0) {
            result += n / 5;
            n /= 5;
        }
        return result;

    }

    public String addBinary(String a, String b) {
        StringBuilder stringBuilder = new StringBuilder();
        int m = a.length() - 1;
        int n = b.length() - 1;
        int carry = 0;
        while (m >= 0 || n >= 0 || carry > 0) {
            int val = (m >= 0 ? Character.getNumericValue(a.charAt(m--)) : 0) + (n >= 0 ? Character.getNumericValue(b.charAt(n--)) : 0) + carry;

            stringBuilder.append(val % 2);

            carry = val / 2;
        }
        if (stringBuilder.length() == 1) {
            return stringBuilder.toString();
        }
        StringBuilder reverse = stringBuilder.reverse();

        return reverse.charAt(0) == '0' ? reverse.substring(1) : reverse.toString();
    }


    // you need treat n as an unsigned value
    public int reverseBits(int n) {
        String reverse = Integer.toBinaryString(n);

        int result = 0;

        for (int i = 0; i < 32 && n != 0; i++) {
            result |= (n & 1) << (31 - i);

            n >>>= 1;
        }
        return result;

    }

    public int hammingWeight(int n) {
        int count = 0;
        while (n != 0) {
            count++;
            n &= (n - 1);
        }
        return count;
    }


    public int singleNumber(int[] nums) {
        int result = 0;
        for (int num : nums) {
            result ^= num;
        }
        return result;
    }


    public int singleNumberii(int[] nums) {
        int low = 0;
        int high = 0;

        for (int n : nums) {
            int carry = low & n;
            low ^= n;
            high |= carry;
            int reset = low ^ high;
            low &= reset;
            high &= reset;
        }

        return low;

    }


    public int findPeakElement(int[] nums) {
        int left = 0;
        int right = nums.length - 1;
        while (left < right) {
            if (nums[left] == nums[right]) {
                right--;
                continue;
            }
            int mid = left + (right - left) / 2;
            if (nums[mid] < nums[mid + 1]) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return left;
    }


    public int search(int[] nums, int target) {
        int left = 0;
        int right = nums.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[left] <= nums[mid]) {
                if (target < nums[mid] && target >= nums[left]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            } else {
                if (target > nums[mid] && target <= nums[right]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }
        return nums[left];
    }

    public void surround(char[][] board) {
        if (board == null || board.length == 0) {
            return;
        }
        int row = board.length;
        int column = board[0].length;
        boolean[][] used = new boolean[row][column];
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (isEdge(i, j, board) && board[i][j] == 'O') {
                    internalSurround(used, i, j, board);
                }
            }
        }
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (board[i][j] == 'o') {
                    board[i][j] = 'O';
                } else if (board[i][j] == 'O') {
                    board[i][j] = 'X';
                }
            }
        }
    }

    private void internalSurround(boolean[][] used, int i, int j, char[][] board) {
        if (i < 0 || i >= board.length || j < 0 || j >= board[0].length) {
            return;
        }
        if (used[i][j]) {
            return;
        }
        used[i][j] = true;
        if (board[i][j] != 'O') {
            return;
        }
        board[i][j] = 'o';
        internalSurround(used, i - 1, j, board);
        internalSurround(used, i + 1, j, board);
        internalSurround(used, i, j - 1, board);
        internalSurround(used, i, j + 1, board);

    }


    private boolean isEdge(int i, int y, char[][] board) {
        return i == 0 || i == board.length - 1 || y == 0 || y == board[0].length - 1;
    }


    public boolean canFinish(int numCourses, int[][] prerequisites) {
        if (prerequisites == null || prerequisites.length == 0) {
            return true;
        }
        List<List<Integer>> graph = new ArrayList<>(numCourses);
        for (int i = 0; i < prerequisites.length; i++) {
            graph.add(new ArrayList<>(prerequisites.length));
        }
        int[] inDegree = new int[numCourses];

        for (int[] prerequisite : prerequisites) {
            int leftEdge = prerequisite[0];
            int rightEdge = prerequisite[1];

            graph.get(rightEdge).add(leftEdge);
            inDegree[leftEdge]++;
        }
        LinkedList<Integer> linkedList = new LinkedList<>();
        for (int i = 0; i < numCourses; i++) {
            if (inDegree[i] == 0) {
                linkedList.offer(i);
            }
        }
        int visitCourse = 0;
        while (!linkedList.isEmpty()) {
            Integer pop = linkedList.pop();
            visitCourse++;

            List<Integer> neighbors = graph.get(pop);

            for (Integer neighbor : neighbors) {
                inDegree[neighbor]--;

                if (inDegree[neighbor] == 0) {
                    linkedList.offer(neighbor);
                }
            }

        }

        return visitCourse == numCourses;

    }

    public List<String> summaryRanges(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new ArrayList<>();
        }
        List<String> result = new ArrayList<>();

        int low = nums[0];

        for (int i = 1; i < nums.length; i++) {
            if (nums[i] != nums[i - 1] + 1) {
                result.add(constructRange(low, nums[i - 1]));
                low = nums[i];
            }
        }
        if (low <= nums[nums.length - 1]) {
            result.add(constructRange(low, nums[nums.length - 1]));
        }
        return result;
    }

    private String constructRange(int start, int end) {
        if (start == end) {
            return String.valueOf(start);
        }
        return start + "->" + end;

    }

    public int calculate(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        }
        int startIndex = 0;
        int len = s.length();
        Stack<Integer> stack = new Stack<>();

        int sign = 1;

        while (startIndex < len) {
            if (s.charAt(startIndex) == '(') {
                int endIndex = startIndex + 1;
                int count = 1;
                while (endIndex < len) {
                    if (Character.isDigit(s.charAt(endIndex))) {
                        endIndex++;
                        continue;
                    }
                    if (s.charAt(endIndex) == '(') {
                        count++;
                    }
                    if (s.charAt(endIndex) == ')') {
                        count--;
                    }
                    if (count == 0) {
                        break;
                    }
                    endIndex++;
                }
                int calculate = calculate(s.substring(startIndex + 1, endIndex));
                stack.push(sign * calculate);
                startIndex = endIndex + 1;
            }

            if (startIndex < len && Character.isDigit(s.charAt(startIndex))) {
                int endIndex = startIndex;
                int tmp = 0;
                while (endIndex < len && Character.isDigit(s.charAt(endIndex))) {
                    tmp = tmp * 10 + Character.getNumericValue(s.charAt(endIndex++));
                }
                tmp = tmp * sign;
                stack.push(tmp);
                startIndex = endIndex;
            }
            if (startIndex < len) {
                if (s.charAt(startIndex) == '+') {
                    sign = 1;
                    startIndex++;
                } else if (s.charAt(startIndex) == '-') {
                    sign = -1;
                    startIndex++;
                } else {
                    startIndex++;
                }
            }
        }
        int result = 0;
        for (Integer number : stack) {
            result += number;
        }
        return result;

    }

    public ListNode deleteDuplicates(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }

        if (head.next.val != head.val) {
            head.next = deleteDuplicates(head.next);
            return head;
        }
        ListNode current = head.next;

        while (current != null && current.val == head.val) {
            current = current.next;
        }
        return deleteDuplicates(current);
    }


    public Node connect(Node root) {

        if (root == null) {
            return root;
        }
//        Node current = root;
//        while (current != null) {
//            Node head = null;
//            Node tmp = null;
//            while (current != null) {
//                if (current.left != null) {
//                    if (head == null) {
//                        head = current.left;
//                        tmp = head;
//                    } else {
//                        tmp.next = current.left;
//                        tmp = tmp.next;
//                    }
//                }
//                if (current.right != null) {
//                    if (head == null) {
//                        head = current.right;
//                        tmp = head;
//                    } else {
//                        tmp.next = current.right;
//                        tmp = tmp.next;
//                    }
//                }
//                current = current.next;
//            }
//            current = head;
//        }
//        return root;
        Node current = root;
        while (current != null) {
            Node nextLevel = null;
            Node nextHead = null;
            while (current != null) {
                if (current.left != null) {
                    if (nextHead == null) {
                        nextHead = current.left;
                        nextLevel = nextHead;
                    } else {
                        nextLevel.next = current.left;
                        nextLevel = nextLevel.next;
                    }
                }
                if (current.right != null) {
                    if (nextHead == null) {
                        nextHead = current.right;
                        nextLevel = nextHead;
                    } else {
                        nextLevel.next = current.right;
                        nextLevel = nextLevel.next;
                    }
                }
                current = current.next;
            }
            current = nextHead;
        }
        return root;

    }


    public static void main(String[] args) {
        TopInterview topInterview = new TopInterview();
        int[] ranges = new int[]{0, 1, 2, 4, 5, 7};
        topInterview.calculate("(1+(4+5+2)-3)+(6+8)");
    }
}
