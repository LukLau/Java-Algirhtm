package org.dora.algorithm.solution.vip;

import org.dora.algorithm.datastructe.Point;
import org.dora.algorithm.datastructe.TreeNode;

import java.util.*;

/**
 * @author dora
 * @date 2019/9/24
 */
public class VipQuestion {

    public static void main(String[] args) {
        VipQuestion vipQuestion = new VipQuestion();

//        int[] nums = new int[]{-2, 0, -1, 3};
        int[][] edges = new int[][]{{0, 1}, {0, 2}, {0, 3}, {1, 4}};
//        int[][] edges = new int[][]{{0, 1}, {1, 2}, {2, 3}, {1, 3}, {1, 4}};
//        vipQuestion.validTree(5, edges);
//        vipQuestion.numberToWords(1234567891);
//        vipQuestion.numSquares(4);
        String pattern = "abab";
        String s = "redblueredblue";

//        vipQuestion.wordPatternMatch(pattern, s);
//        vipQuestion.canWin("+++++");
//        vipQuestion.getHint("1807", "7810");
//        vipQuestion.removeInvalidParentheses("()())()");
        boolean additiveNumber = vipQuestion.isAdditiveNumber("112358");
        System.out.println(additiveNumber);
    }


    /**
     * @param costs: n x 3 cost matrix
     * @return: An integer, the minimum cost to paint all houses
     */
    public int minCost(int[][] costs) {
        // write your code here
        if (costs == null || costs.length == 0) {
            return 0;
        }
        int row = costs.length;
        for (int i = 1; i < row; i++) {
            int min0 = Math.min(costs[i - 1][1], costs[i - 1][2]) + costs[i][0];
            int min1 = Math.min(costs[i - 1][0], costs[i - 1][2]) + costs[i][1];
            int min2 = Math.min(costs[i - 1][0], costs[i - 1][1]) + costs[i][2];

            costs[i][0] = min0;
            costs[i][1] = min1;
            costs[i][2] = min2;
        }
        return Math.min(Math.min(costs[row - 1][0], costs[row - 1][1]), costs[row - 1][2]);
    }


    /**
     * @param preorder: List[int]
     * @return: return a boolean
     */
    public boolean verifyPreorder(int[] preorder) {
        // write your code here
        if (preorder == null || preorder.length == 0) {
            return false;
        }
        Stack<Integer> integerStack = new Stack<>();
        Integer prev = null;
        for (int iterator : preorder) {
            if (prev != null && integerStack.peek() <= prev) {
                return false;
            }

            while (!integerStack.isEmpty() && integerStack.peek() < prev) {
                prev = integerStack.pop();
            }
            integerStack.push(iterator);
        }
        return true;
    }


    public List<String> binaryTreePaths(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        List<String> result = new ArrayList<>();

        internalBinaryTreePaths(result, root, "");

        return result;
    }

    private void internalBinaryTreePaths(List<String> result, TreeNode root, String s) {
        if (root == null) {
            return;
        }
        if (root.left == null && root.right == null) {
            result.add(s.isEmpty() ? String.valueOf(root.val) : s + "->" + root.val);
            return;
        }
        internalBinaryTreePaths(result, root.left, s.isEmpty() ? String.valueOf(root.val) : s + "->" + root.val);
        internalBinaryTreePaths(result, root.right, s.isEmpty() ? String.valueOf(root.val) : s + "->" + root.val);
    }


    /**
     * @param nums:  an array of n integers
     * @param target: a target
     * @return: the number of index triplets satisfy the condition nums[i] + nums[j] + nums[k] < target
     */
    public int threeSumSmaller(int[] nums, int target) {
        // Write your code here
        if (nums == null || nums.length == 0) {
            return 0;
        }
        Arrays.sort(nums);
        int count = 0;
        for (int i = 0; i < nums.length - 2 && nums[i] <= target; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            int start = i + 1;
            int end = nums[nums.length - 1];
            while (start < end) {
                int val = nums[i] + nums[start] + nums[end];

                if (val < target) {
                    count++;
                }

            }
        }
        return count;
    }


    /**
     * @param n: An integer
     * @param edges: a list of undirected edges
     * @return: true if it's a valid tree, or false
     */
    public boolean validTree(int n, int[][] edges) {
        // write your code here
        if (edges == null) {
            return false;
        }
        if (edges.length != n - 1) {
            return false;
        }

        Map<Integer, Set<Integer>> matrix = new HashMap<>();
        for (int[] edge : edges) {
            int leftEdge = edge[0];
            int rightEdge = edge[1];
            Set<Integer> leftNeighbors = matrix.getOrDefault(leftEdge, new HashSet<>());
            Set<Integer> rightNeighbors = matrix.getOrDefault(rightEdge, new HashSet<>());

            leftNeighbors.add(rightEdge);
            rightNeighbors.add(leftEdge);

            matrix.put(leftEdge, leftNeighbors);
            matrix.put(rightEdge, rightNeighbors);
        }
        LinkedList<Integer> linkedList = new LinkedList<>();
        boolean[] visited = new boolean[n];
        int count = 1;
        linkedList.offer(0);
        Set<Integer> used = new HashSet<>();
        visited[0] = true;
//        used.add(0);
        while (!linkedList.isEmpty()) {
            Integer currentNode = linkedList.pop();
            Set<Integer> neighbors = matrix.getOrDefault(currentNode, new HashSet<>());
            for (Integer neighbor : neighbors) {
                if (visited[neighbor]) {
                    continue;
                }
//                if (used.contains(neighbor)) {
//                    continue;
//                }
                count++;
//                used.add(neighbor);
                visited[neighbor] = true;
                linkedList.offer(neighbor);
            }
        }
        return count == n;
    }


    /**
     * 273
     * @param num
     * @return
     */
    public String numberToWords(int num) {
        if (num == 0) {
            return "ZERO";
        }
        String[] one = new String[]{"", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven",
                "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"};
        String[] two = new String[]{"", "Ten", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"};
        String[] three = new String[]{"", "Thousand", "Million", "Billion"};

        int index = 0;
        StringBuilder resultBuilder = new StringBuilder();
        while (num != 0) {
            int remain = num % 1000;
            String value = getValue(remain, one, two, three);

            value = value.trim();

            String first = three[index];

            if (!first.isEmpty() && !value.isEmpty()) {
                value += " " + first + " ";
            }
            resultBuilder.insert(0, value);

            if (num / 1000 != 0) {
                index++;
            }
            num /= 1000;
        }
        return resultBuilder.toString().trim();
    }

    private String getValue(int num, String[] one, String[] two, String[] three) {
        StringBuilder builder = new StringBuilder();

        int first = num / 100;

        if (first != 0) {
            builder.append(one[first]).append(" Hundred");
        }
        int remain = num % 100;

        if (remain < 20) {
            String index = one[remain % 20];
            builder.append(index.isEmpty() ? index : " " + index);
        } else {
            builder.append(" ").append(two[remain / 10]);

            int index = remain % 10;

            String remainValue = one[index];
            builder.append(remainValue.isEmpty() ? remainValue : " " + remainValue);
        }
//        System.out.println(builder);
        return builder.toString();
    }


    /**
     * @param n: non-negative integer, n posts
     * @param k: non-negative integer, k colors
     * @return: an integer, the total number of ways
     */
    public int numWays(int n, int k) {
        // write your code here
        int[] result = new int[n];
        result[0] = k;
        result[1] = k * k;
        for (int i = 2; i < n; i++) {
            result[i] = (k - 1) * (result[i - 1] + result[i - 2]);
        }
        return result[n - 1];
    }

    /**
     * @param n a party with n people
     * @return the celebrity's label or -1
     */
    public int findCelebrity(int n) {
        // Write your code here
        int celebrity = 0;
        for (int i = 1; i < n; i++) {
            if (knows(celebrity, i) || !knows(i, celebrity)) {
                celebrity = i;
            }
        }
        for (int i = 0; i < n; i++) {
            if (i == celebrity) {
                continue;
            }
            if (knows(celebrity, i) || !knows(i, celebrity)) {
                return -1;
            }
        }
        return celebrity;
    }

    private boolean knows(int a, int b) {
        return true;
    }


    public int numSquares(int n) {
        int[] result = new int[n + 1];

        result[0] = 0;
        result[1] = 1;
        for (int i = 2; i <= n; i++) {
            int min = i;
            for (int j = 2; j * j <= i; j++) {
                min = Math.min(min, 1 + result[i - j * j]);
            }
            result[i] = min;
        }
        return result[n];
    }


    /*
     * @param root: The root of the BST.
     * @param p: You need find the successor node of p.
     * @return: Successor of p.
     */
    public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
        // write your code here
        if (root == null) {
            return null;
        }
        Stack<TreeNode> stack = new Stack<>();
        TreeNode prev = root;
        List<Integer> list = new ArrayList<>();
        while (!stack.isEmpty() || prev != null) {
            while (p != null) {
                stack.push(p);
                p = p.left;
            }
            p = stack.pop();
            list.add(p.val);
            p = p.right;
        }


        return null;
    }

    public boolean wordPattern(String pattern, String s) {
        Map<Character, Integer> patternMap = new HashMap<>();
        Map<String, Integer> wordsMap = new HashMap<>();
        String[] words = s.split(" ");
        if (pattern.length() != words.length) {
            return false;
        }
        for (int i = 0; i < words.length; i++) {
            Integer left = patternMap.getOrDefault(pattern.charAt(i), i);

            Integer leftIndex = wordsMap.getOrDefault(words[i], i);

            if (!Objects.equals(left, leftIndex)) {
                return false;
            }
            patternMap.put(pattern.charAt(i), i);
            wordsMap.put(words[i], i);
        }
        return true;

    }


    /**
     * 829 · 字模式 II
     * @param pattern: a string,denote pattern string
     * @param str: a string, denote matching string
     * @return: a boolean
     */
    public boolean wordPatternMatch(String pattern, String str) {
        // write your code here
        if (pattern.isEmpty() && str.isEmpty()) {
            return true;
        }
        if (pattern.isEmpty() || str.isEmpty()) {
            return false;
        }
        Set<String> visited = new HashSet<>();
        return internalWordPatternMatch(new HashMap<>(), visited, pattern, str);
    }

    private boolean internalWordPatternMatch(Map<Character, String> wordMap, Set<String> visited, String pattern, String str) {
        if (pattern.isEmpty() && str.isEmpty()) {
            return true;
        }
        if (pattern.isEmpty() || str.isEmpty()) {
            return false;
        }
        Character firstLetter = pattern.charAt(0);

        if (wordMap.containsKey(firstLetter)) {
            String abbrWord = wordMap.get(firstLetter);

            if (!str.startsWith(abbrWord)) {
                return false;
            }
            return internalWordPatternMatch(wordMap, visited, pattern.substring(1), str.substring(abbrWord.length()));
        }
        for (int i = 1; i <= str.length(); i++) {
            String prefix = str.substring(0, i);

            char letter = pattern.charAt(0);


            if (visited.contains(prefix)) {
                continue;
            }
            wordMap.put(letter, prefix);
            visited.add(prefix);
            if (internalWordPatternMatch(wordMap, visited, pattern.substring(1), str.substring(i))) {
                return true;
            }
            wordMap.remove(letter);
            visited.remove(prefix);
        }
        return false;
    }


    public boolean canWinNim(int n) {
        if (n == 1 || n == 3 || n == 2 || n == 5) {
            return true;
        }
        if (n == 4) {
            return false;
        }
        boolean[] numbers = new boolean[n + 1];
        numbers[1] = true;
        numbers[3] = true;
        numbers[2] = true;
        numbers[5] = true;
        numbers[4] = false;
        for (int i = 6; i <= n; i++) {
            boolean winner = !numbers[i - 1];
            if (!winner && !numbers[i - 2]) {
                winner = true;
            }
            if (!winner && !numbers[i - 3]) {
                winner = true;
            }
            numbers[i] = winner;
        }
        return numbers[n];
    }

    /**
     * @param s: the given string
     * @return: all the possible states of the string after one valid move
     *          we will sort your return value in output
     */
    public List<String> generatePossibleNextMoves(String s) {
        // write your code here
        if (s == null) {
            return new ArrayList<>();
        }
        List<String> result = new ArrayList<>();
        return result;
    }

    /**
     * @param s: the given string
     * @return: if the starting player can guarantee a win
     */
    public boolean canWin(String s) {
        // write your code here
        if (s == null || s.isEmpty()) {
            return true;
        }
        Set<String> visited = new HashSet<>();

        int index = 0;
        int len = s.length();
        while (index <= len - 2) {
            int indexOf = s.indexOf("++", index);
            if (indexOf == -1) {
                return false;
            }
            String prefix = indexOf == 0 ? "" : s.substring(0, indexOf);

            String tmp = prefix + "--" + s.substring(indexOf + 2);

//            if (visited.contains(tmp)) {
//                continue;
//            }

            if (!canWin(tmp)) {
                return true;
            }
            index = indexOf + 1;
        }
        return false;
    }

    /**
     * @param root: the root of binary tree
     * @return: the length of the longest consecutive sequence path
     */
    private int maxLongestLen = 0;

    public int longestConsecutive(TreeNode root) {
        // write your code here
        if (root == null) {
            return 0;
        }
        internalTree(root);
        return maxLongestLen;
    }

    private int internalTree(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int left = internalTree(root.left);
        int right = internalTree(root.right);
        if (root.left == null || root.val + 1 != root.left.val) {
            left = 0;
        }
        if (root.right == null || root.val + 1 != root.right.val) {
            right = 0;
        }
        int result = Math.max(left, right) + 1;

        maxLongestLen = Math.max(maxLongestLen, result);

        return result;
    }


    // todo
    private static int maxLongestLenV2 = 0;

    /**
     * @param root: the root of binary tree
     * @return: the length of the longest consecutive sequence path
     */
    public int longestConsecutive2(TreeNode root) {
        // write your code here
        TreeResult result = helper(root);

        return result.longestResult;
    }

    private TreeResult helper(TreeNode root) {
        if (root == null) {
            return new TreeResult(0, 0, 0);
        }
        TreeResult left = helper(root.left);
        TreeResult right = helper(root.right);


        int leftIncr = left.increase;
        int leftDecr = left.decrease;
        int rightIncr = right.increase;
        int rightDecr = right.decrease;
        int currentIncrease = 0;
        int currentDecrease = 0;

        // 更新当前节点为端点时的最长递增和递减的长度
        if (root.left != null) {
            if (root.val == root.left.val - 1) {
                currentIncrease = Math.max(currentIncrease, leftIncr + 1);
            }
            if (root.val == root.left.val + 1) {
                currentDecrease = Math.max(currentDecrease, leftDecr + 1);
            }
        }

        if (root.right != null) {
            if (root.val == root.right.val - 1) {
                currentIncrease = Math.max(currentIncrease, rightIncr + 1);
            }
            if (root.val == root.right.val + 1) {
                currentDecrease = Math.max(currentDecrease, rightDecr + 1);
            }
        }
        // 更新答案
        int longestLength = Math.max(Math.max(left.longestResult, right.longestResult),
                currentIncrease + currentDecrease + 1);

        // 返回当前节点为端点时的最长递增和递减长度
        return new TreeResult(currentIncrease, currentDecrease, longestLength);


    }

    static class TreeResult {
        private int increase;

        private int decrease;

        private int longestResult;

        public TreeResult(int increase, int decrease, int longestResult) {
            this.increase = increase;
            this.decrease = decrease;
            this.longestResult = longestResult;
        }
    }

    public String getHint(String secret, String guess) {
        int m = secret.length();
        int n = guess.length();
        if (m != n) {
            return "";
        }
        int[] hash = new int[256];
        int bull = 0;
        int cow = 0;
        int index = 0;
        while (index < m) {
            char s = secret.charAt(index);

            char g = guess.charAt(index);

            if (s == g) {
                bull++;
            } else {
                int gIndex = g - '0';
                int sIndex = s - '0';

                if (hash[gIndex]-- > 0) {
                    cow++;
                }
                if (hash[sIndex]++ < 0) {
                    cow++;
                }
            }
            index++;
        }
        return bull + "A" + cow + "B";
    }

    public int lengthOfLIS(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int len = nums.length;
        int[] result = new int[len];
        Arrays.fill(result, 1);
        for (int i = 1; i < len; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[j] < nums[i] && result[j] + 1 > result[i]) {
                    result[i] = result[j] + 1;
                }
            }
        }
        int answer = 0;
        for (int tmp : result) {
            answer = Math.max(answer, tmp);
        }
        return answer;
    }

    public List<String> removeInvalidParentheses(String s) {
        if (s == null || s.isEmpty()) {
            return new ArrayList<>();
        }

        Set<String> visited = new HashSet<>();

        List<String> result = new ArrayList<>();
        LinkedList<String> linkedList = new LinkedList<>();
        linkedList.offer(s);

        while (!linkedList.isEmpty()) {
            if (!result.isEmpty()) {
                break;
            }
            int size = linkedList.size();

            for (int index = 0; index < size; index++) {
                String pop = linkedList.pop();
                if (isValidString(pop)) {
                    result.add(pop);
                    continue;
                }
                int len = pop.length();
                for (int i = 0; i < len; i++) {
                    char current = pop.charAt(i);

                    if (current != '(' && current != ')') {
                        continue;
                    }
                    String tmp = i == 0 ? pop.substring(i + 1) : pop.substring(0, i) + pop.substring(i + 1);

                    if (visited.contains(tmp)) {
                        continue;
                    }
                    visited.add(tmp);
                    linkedList.offer(tmp);
                }
            }


        }
        return new ArrayList<>(result);

    }

    private boolean isValidString(String s) {
        int count = 0;
        int len = s.length();
        for (int i = 0; i < len; i++) {
            char c = s.charAt(i);
            if (c != '(' && c != ')') {
                continue;
            }
            if (count == 0 && c == ')') {
                return false;
            }
            if (c == '(') {
                count++;
            }
            if (c == ')') {
                count--;
            }
        }
        return count == 0;
    }

    /**
     * @param image: a binary matrix with '0' and '1'
     * @param x: the location of one of the black pixels
     * @param y: the location of one of the black pixels
     * @return: an integer
     */
    public int minArea(char[][] image, int x, int y) {
        // write your code here
        if (image == null || image.length == 0) {
            return 0;
        }
        int row = image.length;
        int column = image[0].length;
//        int leftEdge = calculate(image, 0, y, x, true);
//        int rightEdge = calculate(image,y,column-1,x, false);

        // todo
        return -1;
    }

    /**
     * @param n: An integer
     * @param m: An integer
     * @param operators: an array of point
     * @return: an integer array
     */
    public List<Integer> numIslands2(int n, int m, Point[] operators) {
        if (operators == null || operators.length == 0) {
            return new ArrayList<>();
        }
        // write your code here
        List<Integer> result = new ArrayList<>();

        int[][] islands = new int[n][m];
        for (int i = 0; i < operators.length; i++) {
            Point operator = operators[i];
            int currentRow = operator.x;
            int currentColumn = operator.y;
            islands[currentRow][currentColumn] = 1;
            LinkedList<Point> linkedList = new LinkedList<>();
            linkedList.offer(operator);

            int count = 0;
            while (!linkedList.isEmpty()) {
                Point poll = linkedList.poll();
            }
        }
        // todo
        return new ArrayList<>();
    }


    private boolean isInValidEdge(int[][] islands, int x, int y) {
        return x < 0 || x == islands.length || y < 0 || y == islands[x].length;
    }

    public boolean isAdditiveNumber(String num) {
        if (num == null || num.isEmpty()) {
            return true;
        }
        int len = num.length();
        for (int i = 1; i <= len; i++) {
            String prefix = num.substring(0, i);
            if (i > 1 && prefix.charAt(0) == '0') {
                continue;
            }
            String remain = num.substring(i);
            if (internalIsAdditive(remain, prefix)) {
                return true;
            }
        }
        return false;
    }

    private boolean internalIsAdditive(String num, String prefix) {
        if (num.length() <= 1) {
            return false;
        }
        int len = num.length();

        for (int i = 1; i <= len; i++) {
            String predecessor = num.substring(0, i);
            if (i > 1 && predecessor.charAt(0) == '0') {
                continue;
            }
            long previousValue = Long.parseLong(prefix);

            long currentValue = Long.parseLong(predecessor);

            long sum = previousValue + currentValue;

            String remain = num.substring(i);

            String sumValue = String.valueOf(sum);

            if (remain.equals(sumValue)) {
                return true;
            }
            if (remain.startsWith(sumValue) && internalIsAdditive(remain, predecessor)) {
                return true;
            }
        }
        return false;
    }


}

