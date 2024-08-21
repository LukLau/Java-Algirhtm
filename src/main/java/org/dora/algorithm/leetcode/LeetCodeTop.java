package org.dora.algorithm.leetcode;

import org.dora.algorithm.datastructe.Node;
import org.dora.algorithm.datastructe.TreeNode;

import java.util.*;

/**
 * date 2024年04月07日
 */
public class LeetCodeTop {

    public static void main(String[] args) {

        LeetCodeTop leetCodeTop = new LeetCodeTop();

//        int[] kth = new int[]{1, 1, 1, 2, 2, 3, 3, 3};
//        leetCodeTop.topKFrequent(kth, 2);


//        leetCodeTop.rotate(new int[]{1, 2, 3, 4, 5, 6, 7}, 3);
        leetCodeTop.orangesRottingii(new int[][]{{2, 1, 1}, {1, 1, 0}, {0, 1, 1}});
    }

    public List<List<String>> groupAnagrams(String[] strs) {
        if (strs == null || strs.length == 0) {
            return new ArrayList<>();
        }
        Map<String, List<String>> result = new HashMap<>();
        for (String str : strs) {
            char[] words = str.toCharArray();
            Arrays.sort(words);
            String value = String.valueOf(words);
            List<String> list = result.getOrDefault(value, new ArrayList<>());
            list.add(str);


            result.put(value, list);
        }
        return new ArrayList<>(result.values());

    }


    public int longestConsecutive(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        Map<Integer, Integer> map = new HashMap<>();
        int result = 0;
        for (int num : nums) {
            if (!map.containsKey(num)) {
                int left = map.getOrDefault(num - 1, 0);
                int right = map.getOrDefault(num + 1, 0);


                int tmp = left + right + 1;
                result = Math.max(result, tmp);

                map.put(num - left, tmp);
                map.put(num + right, tmp);
                map.put(num, tmp);
            }
        }
        return result;
    }


    public void moveZeroes(int[] nums) {
        if (nums == null || nums.length == 0) {
            return;
        }
        int firstIndex = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != 0) {
                nums[firstIndex++] = nums[i];
            }
        }
        while (firstIndex < nums.length) {
            nums[firstIndex++] = 0;
        }
    }

    public List<List<Integer>> threeSum(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new ArrayList<>();
        }
        Arrays.sort(nums);
        List<List<Integer>> result = new ArrayList<>();

        for (int i = 0; i < nums.length - 2; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            int start = i + 1;
            int end = nums.length - 1;
            while (start < end) {
                int val = nums[i] + nums[start] + nums[end];

                if (val == 0) {
                    List<Integer> tmp = Arrays.asList(nums[i], nums[start], nums[end]);
                    result.add(tmp);
                    while (start < end && nums[start] == nums[start + 1]) {
                        start++;
                    }
                    while (start < end && nums[end] == nums[end - 1]) {
                        end--;
                    }
                    start++;
                    end--;
                } else if (val < 0) {
                    start++;
                } else {
                    end--;
                }
            }
        }
        return result;
    }

    public List<Integer> findAnagrams(String s, String p) {
        List<Integer> result = new ArrayList<>();
        int len = p.length();

        char[] words = p.toCharArray();

        Arrays.sort(words);

        String match = String.valueOf(words);

        for (int i = 0; i <= s.length() - len; i++) {
            String sub = s.substring(i, i + len);

            char[] subCharArray = sub.toCharArray();
            Arrays.sort(subCharArray);

            String s1 = String.valueOf(subCharArray);

            if (s1.equals(match)) {
                result.add(i);
            }
        }
        return result;
    }


    public int[] maxSlidingWindow(int[] nums, int k) {
        if (nums == null || nums.length == 0) {
            return new int[]{};
        }
        List<Integer> result = new ArrayList<>();
        LinkedList<Integer> linkedList = new LinkedList<>();
        for (int i = 0; i < nums.length; i++) {
            int index = i - k + 1;
            while (!linkedList.isEmpty() && nums[linkedList.peekLast()] <= nums[i]) {
                linkedList.pollLast();
            }
            linkedList.offer(i);

            while (!linkedList.isEmpty() && linkedList.peekFirst() < index) {
                linkedList.pollFirst();
            }
            if (index >= 0) {
                result.add(linkedList.peekFirst());
            }
        }
        int[] ans = new int[result.size()];

        for (int i = 0; i < result.size(); i++) {
            ans[i] = nums[result.get(i)];
        }
        return ans;
    }

    public int maxSubArray(int[] nums) {
        int result = Integer.MIN_VALUE;
        int local = 0;
        for (int num : nums) {
            local += num;
            result = Math.max(result, local);

            if (local < 0) {
                local = 0;
            }
        }
        return result;
    }


    public int[] productExceptSelf(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new int[]{};
        }
        int[] result = new int[nums.length];
        int base = 1;
        for (int i = 0; i < nums.length; i++) {
            result[i] = base;
            base *= nums[i];
        }
        base = 1;
        for (int i = nums.length - 1; i >= 0; i--) {
            result[i] *= base;
            base *= nums[i];
        }
        return result;
    }


    public List<Integer> spiralOrder(int[][] matrix) {
        if (matrix == null || matrix.length == 0) {
            return new ArrayList<>();
        }
        List<Integer> result = new ArrayList<>();
        int row = matrix.length;
        int column = matrix[0].length;

        int left = 0;
        int right = column - 1;

        int top = 0;
        int bottom = row - 1;
        while (left <= right && top <= bottom) {
            for (int i = left; i <= right; i++) {
                result.add(matrix[top][i]);
            }
            for (int i = top + 1; i <= bottom; i++) {
                result.add(matrix[i][right]);
            }
            if (top != bottom) {
                for (int i = right - 1; i >= left; i--) {
                    result.add(matrix[bottom][i]);
                }
            }
            if (left != right) {
                for (int i = bottom - 1; i > top; i--) {
                    result.add(matrix[i][left]);
                }
            }
            left++;
            right--;
            top++;
            bottom--;
        }
        return result;

    }


    public boolean searchMatrixii(int[][] matrix, int target) {
        if (matrix == null || matrix.length == 0) {
            return false;
        }
        int row = matrix.length;
        int column = matrix[0].length;
        int i = 0;
        int j = column - 1;
        while (i < row && j >= 0) {
            int val = matrix[i][j];
            if (val == target) {
                return true;
            } else if (val < target) {
                i++;
            } else {
                i--;
            }
        }
        return false;

    }


    public Node copyRandomList(Node head) {
        if (head == null) {
            return null;
        }
        Node current = head;
        while (current != null) {
            Node tmp = new Node(current.val);


            Node next = current.next;

            tmp.next = next;

            current.next = tmp;

            current = next;
        }

        current = head;

        while (current != null) {
            Node random = current.random;

            if (random != null) {
                current.next.random = random.next;
            }
            current = current.next.next;
        }

        current = head;

        Node phead = current.next;

        while (current.next != null) {
            Node tmp = current.next;
            current.next = tmp.next;
            current = tmp;
        }
        return phead;

    }

    public TreeNode invertTree(TreeNode root) {
        if (root == null) {
            return root;
        }
        TreeNode left = root.left;
        TreeNode right = root.right;

        root.left = right;
        root.right = left;

        invertTree(left);
        invertTree(right);
        return root;
    }


    public List<List<Integer>> levelOrder(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        LinkedList<TreeNode> linkedList = new LinkedList<>();

        linkedList.offer(root);

        List<List<Integer>> result = new ArrayList<>();

        while (!linkedList.isEmpty()) {

            int size = linkedList.size();

            List<Integer> tmp = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                TreeNode node = linkedList.poll();

                tmp.add(node.val);

                if (node.left != null) {
                    linkedList.offer(node.left);
                }
                if (node.right != null) {
                    linkedList.offer(node.right);
                }

            }
            result.add(tmp);
        }
        return result;
    }


    public int kthSmallest(TreeNode root, int k) {
        Stack<TreeNode> stack = new Stack<>();

        TreeNode p = root;

        int count = 0;
        while (!stack.isEmpty() || p != null) {
            while (p != null) {
                stack.push(p);
                p = p.left;
            }
            p = stack.pop();

            count++;
            if (count == k) {
                return p.val;
            }
        }
        return -1;
    }


    public List<Integer> rightSideView(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        Map<Integer, Integer> depthMap = new HashMap<>();

        LinkedList<Integer> linkedList = new LinkedList<>();
        LinkedList<TreeNode> nodeList = new LinkedList<>();

        linkedList.offer(0);
        nodeList.offer(root);

        while (!linkedList.isEmpty()) {
            Integer depth = linkedList.poll();
            TreeNode node = nodeList.poll();

            if (!depthMap.containsKey(depth)) {
                depthMap.put(depth, node.val);
            }
            if (node.right != null) {
                nodeList.offer(node.right);
                linkedList.offer(depth + 1);
            }
            if (node.left != null) {
                nodeList.offer(node.left);
                linkedList.offer(depth + 1);
            }
        }
        List<Integer> result = new ArrayList<>();
        for (Map.Entry<Integer, Integer> item : depthMap.entrySet()) {
            Integer value = item.getValue();
            result.add(value);
        }
        return result;
    }

    public int pathSum(TreeNode root, int targetSum) {
        if (root == null) {
            return 0;
        }
        int count = internalPathSum(root, targetSum);
        count += pathSum(root.left, targetSum);
        count += pathSum(root.right, targetSum);
        return count;
    }

    private int internalPathSum(TreeNode root, long targetSum) {
        if (root == null) {
            return 0;
        }
        int count = 0;
        if (root.val == targetSum) {
            count++;
        }
        count += internalPathSum(root.left, targetSum - root.val);
        count += internalPathSum(root.right, targetSum - root.val);
        return count;
    }

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null) {
            return null;
        }
        if (root == p || root == q) {
            return root;
        }
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if (left != null && right != null) {
            return p;
        } else if (left != null) {
            return left;
        } else {
            return right;
        }
    }

    private int maxPathResult = 0;

    public int maxPathSum(TreeNode root) {
        if (root == null) {
            return 0;
        }
        internalMaxPathSum(root, 0);

        return maxPathResult;

    }

    private void internalMaxPathSum(TreeNode root, int value) {
        if (root == null) {
            return;
        }
        value += root.val;


        this.maxPathResult = Math.max(maxPathResult, value);

    }

    public int findMin(int[] nums) {
        int left = 0;
        int right = nums.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] > nums[right]) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return nums[left];

    }

    public int largestRectangleArea(int[] heights) {
        if (heights == null || heights.length == 0) {
            return 0;
        }
        Stack<Integer> stack = new Stack<>();
        int result = 0;
        for (int i = 0; i <= heights.length; i++) {
            int side = i == heights.length ? 0 : heights[i];

            if (stack.isEmpty() || heights[stack.peek()] <= side) {
                stack.push(i);
            } else {
                Integer leftSide = stack.pop();

                int width = stack.isEmpty() ? i : i - stack.peek();

                result = Math.max(result, width * heights[leftSide]);

                i--;

            }

        }
        return result;
    }


    public int coinChange(int[] coins, int amount) {
        if (coins == null || coins.length == 0) {
            return -1;
        }
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, Integer.MAX_VALUE);
        dp[0] = 0;
        for (int i = 1; i <= amount; i++) {
            int min = Integer.MAX_VALUE;
            for (int money : coins) {
                if (i - money >= 0 && dp[i - money] != Integer.MAX_VALUE) {
                    min = Math.min(min, dp[i - money] + 1);
                }
            }
            dp[i] = min;
        }
        return dp[amount] == Integer.MAX_VALUE ? -1 : dp[amount];

    }

    public int change(int amount, int[] coins) {
        if (coins == null || coins.length == 0) {
            return 0;
        }
        int[] dp = new int[amount + 1];
        dp[0] = 1;
        for (int coin : coins) {
            for (int i = coin; i <= amount; i++) {
                dp[i] += dp[i - coin];
            }
        }
        return dp[amount];
    }


    // 背包问题


    /**
     * @param m: An integer m denotes the size of a backpack
     * @param a: Given n items with size A[i]
     * @return: The maximum size
     */
    public int backPack(int m, int[] a) {
        // write your code here
        if (a == null || a.length == 0) {
            return 0;
        }
        int[][] result = new int[a.length + 1][m + 1];

        for (int i = 1; i <= a.length; i++) {
            for (int j = 0; j <= m; j++) {
                if (j >= a[i - 1]) {
                    result[i][j] = Math.max(result[i - 1][j], result[i - 1][j - a[i - 1]] + a[i - 1]);
                } else {
                    result[i][j] = result[i - 1][j];
                }
            }
        }
        return result[a.length][m];
    }


    /**
     * @param m: An integer m denotes the size of a backpack
     * @param a: Given n items with size A[i]
     * @param v: Given n items with value V[i]
     * @return: The maximum value
     */
    public int backPackII(int m, int[] a, int[] v) {
        // write your code here
        int[][] result = new int[a.length + 1][m + 1];

        for (int i = 1; i <= a.length; i++) {
            for (int j = 1; j <= m; j++) {
                if (a[i - 1] > j) {
                    result[i][j] = result[i - 1][j];
                } else {
                    result[i][j] = Math.max(result[i - 1][j], result[i - 1][j - a[i - 1]] + v[i - 1]);
                }
            }
        }
        return result[a.length][m];
    }


    /**
     * @param a: an integer array
     * @param v: an integer array
     * @param m: An integer
     * @return: an array
     */
    public int backPackIII(int[] a, int[] v, int m) {
        // write your code here
//        int n = a.length;
//
//        //1.
//        int[][] f = new int[n + 1][m + 1];
//
//        //3.
//        f[0][0] = 0;
//        for (int i = 1; i <= m; i++) {
//            f[0][i] = -1;
//        }
//
//        //2.
//        for (int i = 1; i <= n; i++) {
//            for (int j = 0; j <= m; j++) {
//                // optimize piece
//                f[i][j] = f[i - 1][j];
//                if (j - a[i - 1] >= 0) {
//                    f[i][j] = Math.max(f[i][j - a[i - 1]] + v[i - 1], f[i][j]);
//                }
//                // optimize piece
//            }
//        }
//
//        int res = Integer.MIN_VALUE;
//        for (int w = 0; w <= m; w++) {
//            if (res < f[n][w]) {
//                res = f[n][w];
//            }
//        }
//
//        return res;

        int n = a.length;
        int[][] result = new int[n + 1][m + 1];


        for (int i = 1; i <= n; i++) {
            for (int j = 0; j <= m; j++) {
                if (j - a[i - 1] >= 0) {
                    result[i][j] = Math.max(result[i][j - a[i - 1]] + v[i - 1], result[i - 1][j]);
                } else {
                    result[i][j] = result[i - 1][j];
                }
            }
        }
        return result[n][m];

    }


    /**
     * @param nums: an integer array and all positive numbers, no duplicates
     * @param target: An integer
     * @return: An integer
     */
    public int backPackIV(int[] nums, int target) {
        // write your code here
//        if (coins == null || coins.length == 0) {
//            return 0;
//        }
//        int[] dp = new int[amount + 1];
//        dp[0] = 1;
//        for (int coin : coins) {
//            for (int i = coin; i <= amount; i++) {
//                dp[i] += dp[i - coin];
//            }
//        }
//        return dp[amount];
        int n = nums.length;
        int[] result = new int[target + 1];

        result[0] = 1;

        for (int num : nums) {
            for (int j = num; j <= target; j++) {
                result[j] += result[j - num];
            }
        }
        return result[target];
    }

    /**
     * @param nums: an integer array and all positive numbers
     * @param target: An integer
     * @return: An integer
     */
    public int backPackV(int[] nums, int target) {
        // write your code here
        int[] result = new int[target + 1];
        for (int num : nums) {
            for (int j = 1; j <= target; j++) {
                if (j - num >= 0) {
                    result[j] = Math.min(result[j - num] + 1, result[j]);
                }
            }
        }
        return result[target];
    }


    public int findKthLargest(int[] nums, int k) {
        PriorityQueue<Integer> priorityQueue = new PriorityQueue<>(new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return o1.compareTo(o2);
            }
        });

        for (int num : nums) {
            if (priorityQueue.isEmpty() || priorityQueue.size() < k) {
                priorityQueue.offer(num);
            } else {
                if (num > priorityQueue.peek()) {
                    priorityQueue.poll();
                    priorityQueue.offer(num);
                }
            }
        }
        return priorityQueue.peek();

    }


    public int[] topKFrequent(int[] nums, int k) {
        if (nums == null || nums.length == 0) {
            return new int[]{};
        }
        Map<Integer, Integer> map = new HashMap<>();

        PriorityQueue<Integer> linkedList = new PriorityQueue<>(new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return map.get(o1) - map.get(o2);
            }
        });

        for (int num : nums) {
            Integer count = map.getOrDefault(num, 0);

            count++;

            map.put(num, count);
        }
        for (Map.Entry<Integer, Integer> item : map.entrySet()) {
            Integer num = item.getKey();

            Integer count = item.getValue();

            if (linkedList.isEmpty() || linkedList.size() < k) {
                linkedList.offer(num);
            } else if (count > map.get(linkedList.peek())) {
                linkedList.poll();
                linkedList.offer(num);
            }

        }
        int[] result = new int[linkedList.size()];
        int index = 0;
        for (Integer i : linkedList) {
            result[index++] = i;

        }
        return result;
    }

    private int partition(int[] nums, int start, int end, int target) {
        int pivot = nums[start];
        while (start < end) {
            while (start < end && nums[end] >= pivot) {
                end--;
            }

            if (start < end) {
                swap(nums, start, end);
                start++;
            }
            while (start < end && nums[start] <= pivot) {
                start++;
            }

            if (start < end) {
                swap(nums, start, end);
                end--;
            }
        }
        nums[start] = pivot;

        return start;
    }

    private void swap(int[] nums, int i, int j) {
        int val = nums[i];
        nums[i] = nums[j];
        nums[j] = val;
    }


    public int maxProfit(int[] prices) {
        if (prices == null || prices.length == 0) {
            return 0;
        }
        int cost = prices[0];
        int result = 0;
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > cost) {
                result = Math.max(result, prices[i] - cost);
            } else {
                cost = prices[i];
            }
        }
        return result;
    }


    public int longestCommonSubsequence(String text1, String text2) {
        int m = text1.length();

        int n = text2.length();

        int[][] result = new int[m + 1][n + 1];

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (text1.charAt(i - 1) == text2.charAt(j - 1)) {
                    result[i][j] = 1 + result[i - 1][j - 1];
                } else {
                    result[i][j] = Math.max(result[i - 1][j], result[i][j - 1]);
                }
            }
        }
        return result[m][n];
    }

    public int maxArea(int[] height) {
        if (height == null || height.length == 0) {
            return 0;
        }
        int left = 0;
        int right = height.length - 1;
        int result = 0;
        while (left < right) {
            int val = (Math.max(height[left], height[right])) * (right - left + 1);

            result = Math.max(result, val);

            if (height[left] < height[right]) {
                left++;
            } else {
                right--;
            }
        }
        return result;

    }


    public void rotate(int[] nums, int k) {
        if (nums == null || nums.length == 0) {
            return;
        }
        k %= nums.length;

        if (k == 0) {
            return;
        }
        int i = 0;
        int end = nums.length - 1;

        while (i < end) {
            swap(nums, i, end);
            i++;
            end--;
        }
        i = 0;
        int pivot = k - 1;
        int remain = k;
        while (i < pivot) {
            swap(nums, i, pivot);
            pivot--;
            i++;
        }
        end = nums.length - 1;
        while (remain < end) {
            swap(nums, remain, end);
            remain++;
            end--;
        }
    }


    /**
     *
     * @param s
     * @return
     */
    public String decodeString(String s) {
        if (s == null || s.isEmpty()) {
            return "";
        }
        return "";
    }

    public int orangesRotting(int[][] grid) {
        if (grid == null || grid.length == 0) {
            return 0;
        }
        int row = grid.length;
        int column = grid[0].length;
        int[][] result = new int[row][column];
        for (int i = 0; i < row; i++) {
            Arrays.fill(result[i], row * column);
        }
//        int[][] matrix = new int[][]{{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (grid[i][j] == 2) {
                    internalOrange(0, i, j, grid, result);
                }
            }
        }
        int ans = 0;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (grid[i][j] != 0) {
                    ans = Math.max(result[i][j], ans);
                }
            }
        }
        return ans == row * column ? -1 : ans;
    }

    public void internalOrange(int depth, int i, int j, int[][] grid, int[][] result) {
        if (i < 0 || i >= grid.length || j < 0 || j >= grid[i].length || grid[i][j] == 0 || depth >= result[i][j]) {
            return;
        }
        result[i][j] = Math.min(depth, result[i][j]);
        internalOrange(depth + 1, i - 1, j, grid, result);
        internalOrange(depth + 1, i + 1, j, grid, result);
        internalOrange(depth + 1, i, j - 1, grid, result);
        internalOrange(depth + 1, i, j + 1, grid, result);
    }

    public int orangesRottingii(int[][] grid) {
        if (grid == null || grid.length == 0) {
            return 0;
        }
        LinkedList<int[]> linkedList = new LinkedList<>();

        int row = grid.length;
        int column = grid[0].length;

        int freshNum = 0;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (grid[i][j] == 1) {
                    freshNum++;
                } else if (grid[i][j] == 2) {
                    linkedList.offer(new int[]{i, j});
                }
            }
        }

        int[][] matrix = new int[][]{{0, 1}, {0, -1}, {-1, 0}, {1, 0}};
        int minute = 0;
        while (!linkedList.isEmpty()) {
            int size = linkedList.size();
            if (freshNum == 0) {
                return minute;
            }
            minute++;
            for (int i = 0; i < size; i++) {
                int[] currentPoint = linkedList.poll();
                for (int[] direction : matrix) {
                    int x = currentPoint[0] + direction[0];
                    int y = currentPoint[1] + direction[1];
                    if (x < 0 || x >= row || y < 0 || y >= column || grid[x][y] != 1) {
                        continue;
                    }
                    grid[x][y] = 2;
                    freshNum--;
                    linkedList.offer(new int[]{x, y});
                }
            }
        }
        return freshNum > 0 ? -1 : minute;

    }


}
