package org.learn.algorithm.leetcode;

import org.learn.algorithm.datastructure.ListNode;
import org.learn.algorithm.datastructure.TreeNode;
import org.slf4j.event.SubstituteLoggingEvent;
import org.springframework.web.context.request.NativeWebRequest;

import java.util.*;

/**
 * 第三页
 *
 * @author luk
 * @date 2021/4/13
 */
public class ThreePage {

    public static void main(String[] args) {
        ThreePage page = new ThreePage();
        int[] nums = new int[]{3, 2, 3, 1, 2, 4, 5, 5, 6};
//        page.containsNearbyDuplicate(new int[]{1, 2, 3, 1, 2, 3}, 2);
        int[] tmp = new int[]{0, 1, 2, 4, 5, 7};
//        page.summaryRanges(tmp);
//        page.containsNearbyDuplicate(new int[]{1, 2, 3, 1, 2, 3}, 2);
//        page.isAnagram("anagram", "nagaram");
//        System.out.println(page.isIsomorphic("egg", "add"));
//        nums = new int[]{2, 3, 1, 2, 4, 3};
//        page.minSubArrayLenii(7, nums);


        nums = new int[]{237384, 348185, 338816, 825359, 461215, 315112, 170091, 173609, 724297, 828355, 395935, 687922, 127118, 795454, 166794, 888047, 274616, 667202, 772520, 845914, 142832, 575527, 980196, 584061, 784366, 68694, 80105, 618370, 915739, 787506, 379362, 601205, 44762, 969018, 625507, 738640, 900407, 43638, 477963, 233716, 613792, 751212, 231136, 467042, 514654, 521610, 369778, 843173, 257148, 760274, 234955, 265546, 891429, 750091, 942435, 882691, 485058, 792360, 435287, 372065, 396852, 330275, 801939, 200914, 455728, 109280, 527003, 303337, 793015, 667734, 279361, 37810, 783709, 960454, 33515, 263864, 624043, 788379, 449351, 538850, 706217, 238925, 353546, 816968, 654562, 564565, 948428, 739868, 861656, 857587, 418080, 653483, 909732, 952407, 927615, 35267, 732179, 232870, 933798, 61900, 147786, 777833, 294447, 441553, 819649, 999015, 523917, 507135, 9098, 588245, 831300, 993003, 945127, 295861, 806452, 771032, 921668, 596114, 58389, 569490, 199237, 885193, 713430, 997040, 349454, 18832, 365963, 558789, 793979, 254667, 902665, 558387, 193214, 22187, 589600, 12593, 959246, 288997, 790510, 739049, 415542, 64949, 504677, 744804, 344632, 973744, 984968, 641758, 44835, 276866, 533525, 98170, 362427, 158920, 761852, 350430, 378032, 170007, 937691, 243911, 35862, 531400, 478419, 649378, 146626, 88512, 280222, 670261, 941343, 38126, 514239, 239530, 502593, 525678, 583466, 591214, 539578, 787397, 941549, 774755, 828458, 408562, 867366, 117131, 114317, 466268, 629942, 971386, 113065, 796098, 875547, 527563, 539137, 993188, 177348, 731079, 452588, 788027, 482866, 183936, 891182, 14590, 796156, 74952, 512718, 988078, 316782, 432209, 405396, 909257, 659880, 642669, 784593, 862734, 581531, 297746, 379112, 965834, 757501, 134934, 262876, 151414, 341459, 68737, 335705, 108420, 175807, 49204, 418698, 817329, 683802, 860609, 970110, 562200, 491152, 474457, 82314, 164791, 579463, 324083, 114843, 523863, 305404, 361637, 631095, 644669, 990028, 661638, 98042, 785313, 92497, 393902, 493634, 54193, 786608, 157915, 907705, 900047, 733377, 104396, 800956, 611681, 900941, 905797, 918430, 595666, 611837, 661676, 485663, 356116, 62184, 22936, 101311, 174347, 176398, 692221, 924783, 332203, 23632, 778317, 433707, 631680, 847304, 963864, 713851, 549636, 191334, 907916, 47933, 193726, 839680, 126830, 969619, 628862, 971186, 734295, 940422, 873838, 710272, 209108, 23584, 415483, 344786, 442615, 533638, 869735, 327330, 64692, 223777, 613179, 926166, 278178, 42026, 695892, 190137, 520649, 212425, 591392, 953797, 870512, 920351, 832099, 272207, 808275, 724503, 127489, 522614, 56839, 59895, 102922, 200728, 277140, 756415, 619144, 202689, 448065, 169987, 854403, 718299, 290852, 184409, 866904, 468376, 434288, 683022, 468668, 521252, 630787, 910353, 142102, 776763, 402397, 997568, 460132, 591549, 560002, 140986, 393106, 315836, 299789, 11157, 948822, 736508, 906511, 359633, 102634, 758603, 805615, 991907, 260305, 566652, 360444, 640556, 683744, 490938, 70267, 974394, 483178, 140278, 963198, 607682, 209876, 545401, 940834, 353171, 46725, 959406, 354607, 105717, 363535, 16577, 802543, 148157, 857807, 853261, 1498, 894257, 545218, 888366, 440977, 760701, 96299, 194450, 556914, 761520, 613246, 137907, 808247, 716421, 37096, 466670, 750120, 55193, 913440, 8276, 447656, 578312, 259763, 531876, 512665, 141442, 939404, 522308, 727677, 66404, 306827, 611378, 640320, 463646, 23442, 814541, 819520, 700355, 999156, 299221, 300487, 694583, 925423, 942388, 200218, 951988, 223828, 625765, 806161, 775098, 668188, 354891, 941824, 874533, 790457, 17266, 587582, 782052, 543260, 761066, 33671, 282287, 36517, 641769, 385319, 788130, 444800, 151440, 760399, 156927, 810160, 399433, 20665, 789379, 245026, 869278, 79201, 525654, 154337, 153772, 520742, 409817, 351982, 632677, 27469, 461968, 67163, 598474, 413742, 162730, 398958, 301268, 188138, 454016, 685406, 65119, 424960, 818292, 500147, 531479, 320450, 31786, 847222, 17718, 573327, 675570, 627627, 134531, 278662, 882290, 91973, 715585, 416932, 141758, 330628, 608707, 961155, 644588, 685130, 558779, 690500, 576750, 57187, 662322, 665426, 878670, 398251, 851778, 73445, 291259, 667672, 879137, 421584, 252470, 981007, 959007, 53698, 133588, 258920, 681480, 73704, 369061, 257507, 783989, 955086, 78221, 878089, 393338, 909013, 941463, 497392, 775208, 464141, 885513, 194241, 576540, 994311, 430940, 463882, 476639, 97844, 412750, 31014, 834638, 766103, 636265, 574729, 830894, 789695, 824714, 226564, 302451, 431214, 698974, 268952, 909138, 568967, 526732, 990610, 420963, 887778, 652540, 435751, 931021, 543446, 232181, 236984, 751446, 608881, 804842, 538843, 329712, 355717, 782681, 473005, 767954, 470918, 742657, 728655, 365915, 904237, 846747, 6298, 705289, 457213, 288518, 679537, 486580, 99128, 173621, 770845, 958054, 602720, 628137, 857349, 902228, 143259, 788408, 349345, 704796, 603484, 480846, 169475, 888264, 7467, 101509, 944655, 550048, 233987, 97958, 211933, 262737, 696890, 306842, 268402, 823675, 462983, 500079, 337185, 779420, 235971, 618001, 852936, 348159, 585270, 253322, 133651, 275390, 815377, 859521, 719221, 37242, 211276, 151229, 636060, 782778, 520680, 520329, 485325, 259013, 617869, 511118, 52820, 606179, 457781, 861353, 410867, 169794, 503806, 854135, 485535, 34892, 964607, 585208, 638485, 609536, 731747, 715170, 318192, 802739, 356387, 170816, 357610, 891098, 977234, 976964, 264819, 525494, 418773, 26864, 134949, 843984, 324369, 726637, 41795, 512710, 401258, 835971, 73776, 630372, 381524, 305245, 549503, 725983, 342165, 317492, 850053, 764863, 120577, 43205, 843143, 996410, 732112, 282227, 857503, 197508, 483094, 483191, 404547, 689744, 533489, 54389, 816657, 900804, 547343, 766481, 89532, 836491, 571400, 361547, 453296, 487986, 777471, 581869, 517966, 657816, 205850, 484738, 437893, 369736, 329489, 129082, 995024, 910156, 562018, 690831, 772881, 471961, 68259, 868647, 162362, 698669, 706968, 803864, 200166, 791470, 779783, 656081, 432804, 101615, 903098, 603139, 243133, 63039, 723923, 706235, 176627, 421722, 156099, 251079, 570462, 387579, 135309, 254221, 900525, 401696, 545242, 649953, 265465, 478224, 210210, 870963, 11204, 906806, 593938, 308828, 839903, 344986, 552973, 868191, 423898, 841030, 31248, 201878, 908361, 938518, 702414, 927371, 187915, 582150, 250622, 853272, 378462, 750728, 841296, 646057, 86613, 903991, 884906, 370017, 62265, 971984, 411031, 64080, 252978, 73118, 398306, 312366, 58337, 774908, 11239, 137947, 738056, 492168, 116364, 24769, 96167, 192952, 142183, 517522, 978015, 944570, 354291, 42020, 729306, 258838, 402630, 687489, 965651, 456761, 261371, 7505, 994268, 758338, 292010, 750701, 193464, 514023, 303629, 973093, 998924, 693448, 114160, 989360, 30856, 262046, 625396, 142208, 76524, 64194, 932588, 28579, 606417, 300349, 605278, 223074, 298180, 775329, 468490, 652937, 990775, 698051, 584527, 259521, 53062, 300194, 522713, 38368, 473798, 940083, 295847, 514003, 243339, 886679, 558418, 48354, 266870, 101180, 725339, 414568, 438067, 74220, 606737, 674479, 136313, 758031, 941488, 506637, 146915, 500431, 56167, 876539, 911115, 819750, 526974, 136156, 809654, 819404, 994640, 114101, 610563, 365490, 210767, 558064, 868269, 621800, 943069, 84813, 577138, 794535, 644600, 760143, 357944, 229363, 240759, 31577, 178432, 281788, 921022, 888824, 619432, 71886, 313585, 988663, 434824, 385765, 135830, 842738, 370217, 578942, 354150, 32117, 347770, 436059, 109040, 955493, 608239, 501982, 465481, 519878, 96756, 397057, 209517, 402619, 619978, 231284, 399712, 160156, 531674, 41788, 899754, 535660, 334515, 908049, 679252, 825448, 396682, 349157, 921979, 17598, 811584, 849117, 415521, 968473, 119924, 241966, 956841, 819813, 256247, 926526, 8530, 456200, 933361, 282325, 705205, 553214, 906451, 944192, 889932, 164576, 516882, 859439, 52687, 111709, 778327, 369781, 474722, 264710, 537749, 420656, 227498, 913284, 444034, 86666, 71542};

//        page.containsDuplicateii(nums);
    }


    /**
     * 202. Happy Number
     */
    public boolean isHappy(int n) {
        List<Integer> result = new ArrayList<>();
        while (true) {
            int tmp = 0;
            while (n != 0) {
                int remain = n % 10;
                tmp += remain * remain;
                n /= 10;
            }
            if (tmp == 1) {
                return true;
            }
            if (result.contains(tmp)) {
                return false;
            }
            result.add(tmp);
            n = tmp;
        }
    }


    /**
     * 203. Remove Linked List Elements
     *
     * @param head
     * @param val
     * @return
     */
    public ListNode removeElements(ListNode head, int val) {
        if (head == null) {
            return null;
        }
        if (head.val == val) {
            return removeElements(head.next, val);
        }
        head.next = removeElements(head.next, val);
        return head;
    }


    /**
     * 205. Isomorphic Strings
     *
     * @param s
     * @param t
     * @return
     */
    public boolean isIsomorphic(String s, String t) {
        if (s == null || t == null) {
            return false;
        }
        int m = s.length();
        int n = t.length();
        if (m != n) {
            return false;
        }
        Map<Character, Integer> map1 = new HashMap<>();
        Map<Character, Integer> map2 = new HashMap<>();

        for (int i = 0; i < m; i++) {
            int first = map1.getOrDefault(s.charAt(i), i);
            int second = map1.getOrDefault(t.charAt(i), i);

            if (first != second) {
                return false;
            }
            map1.put(s.charAt(i), i + 1);
            map2.put(t.charAt(i), i + 1);
        }
        return true;
    }


    /**
     * 206. Reverse Linked List
     *
     * @param head
     * @return
     */
    public ListNode reverseList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode reverseList = reverseList(head.next);
        head.next.next = head;
        head.next = null;
        return reverseList;
    }


    /**
     * https://leetcode.com/problems/minimum-size-subarray-sum/solution/
     * todo use OlogN
     * 209. Minimum Size Subarray Sum
     *
     * @param s
     * @param nums
     * @return
     */
    public int minSubArrayLen(int s, int[] nums) {
        return -1;
    }

    public int minSubArrayLenii(int target, int[] nums) {
        int result = Integer.MAX_VALUE;

        int left = 0;

        int sum = 0;

        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];
            while (sum >= target) {
                result = Math.min(result, i - left + 1);

                sum -= nums[left];
                left++;
            }
        }
        return result == Integer.MAX_VALUE ? 0 : result;
    }

    /**
     * 215. Kth Largest Element in an Array
     *
     * @param nums
     * @param k
     * @return
     */
    public int findKthLargest(int[] nums, int k) {
        if (nums == null || nums.length == 0) {
            return -1;
        }

        int index = nums.length - k;
        int partition = getPartition(nums, 0, nums.length - 1);
        while (partition != index) {
            if (partition < index) {
                partition = getPartition(nums, partition + 1, nums.length - 1);
            } else {
                partition = getPartition(nums, 0, partition - 1);
            }
        }
        return nums[partition];
    }

    private int getPartition(int[] nums, int start, int end) {
        int pivot = nums[start];
        while (start < end) {
            while (start < end && nums[end] >= pivot) {
                end--;
            }
            if (start < end) {
                swap(nums, start, end);
            }
            while (start < end && nums[start] <= pivot) {
                start++;
            }
            if (start < end) {
                swap(nums, start, end);
            }
        }
        nums[start] = pivot;
        return start;
    }

    private void swap(int[] nums, int start, int end) {
        int val = nums[start];
        nums[start] = nums[end];
        nums[end] = val;
    }


    /**
     * 217. Contains Duplicate
     *
     * @param nums
     * @return
     */
    public boolean containsDuplicate(int[] nums) {
        if (nums == null || nums.length == 0) {
            return false;
        }
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            if (map.containsKey(num)) {
                System.out.println(num);
                return true;
            }
            map.put(num, 1);
        }
        return false;
    }

    /**
     * 219. Contains Duplicate II
     *
     * @param nums
     * @param k
     * @return
     */
    public boolean containsNearbyDuplicate(int[] nums, int k) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            int val = nums[i];
            Integer previous = map.put(val, i);
            if (previous != null && Math.abs(i - previous) <= k) {
                return true;
            }
        }
        return false;
    }


    /**
     * todo
     * 220. Contains Duplicate III
     *
     * @param nums
     * @param k
     * @param t
     * @return
     */
    public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
        return false;
    }

    /**
     * 222. Count Complete Tree Nodes
     *
     * @param root
     * @return
     */
    public int countNodes(TreeNode root) {
        if (root == null) {
            return 0;
        }
        return 1 + countNodes(root.left) + countNodes(root.right);
    }


    /**
     * todo
     * 223. Rectangle Area
     *
     * @param A
     * @param B
     * @param C
     * @param D
     * @param E
     * @param F
     * @param G
     * @param H
     * @return
     */
    public int computeArea(int A, int B, int C, int D, int E, int F, int G, int H) {

        return -1;
    }


    public TreeNode invertTree(TreeNode root) {
        if (root == null) {
            return null;
        }
        TreeNode left = root.left;
        root.left = root.right;
        root.right = left;
        invertTree(root.left);
        invertTree(root.right);
        return root;
    }


    /**
     * 228. Summary Ranges
     *
     * @param nums
     * @return
     */
    public List<String> summaryRanges(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new ArrayList<>();
        }
        List<String> result = new ArrayList<>();
        Integer prev = null;
        for (int i = 0; i < nums.length; i++) {
            if (i > 0 && nums[i] != nums[i - 1] + 1) {
                result.add(range(nums[prev], nums[i - 1]));
                prev = i;
            }
            if (prev == null) {
                prev = i;
            }
        }
        result.add(range(nums[prev], nums[nums.length - 1]));
        return result;
    }

    private String range(int start, int end) {
        return start == end ? String.valueOf(start) : start + "->" + end;
    }


    /**
     * 234. Palindrome Linked List
     *
     * @param head
     * @return
     */
    public boolean isPalindrome(ListNode head) {
        if (head == null || head.next == null) {
            return true;
        }
        ListNode fast = head;
        ListNode slow = head;
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        ListNode next = slow.next;

        slow.next = null;

        ListNode reverseList = reverseList(next);

        while (head != null && reverseList != null) {
            if (head.val != reverseList.val) {
                return false;
            }
            head = head.next;
            reverseList = reverseList.next;
        }
        return true;
    }


    /**
     * 237. Delete Node in a Linked List
     *
     * @param node
     */
    public void deleteNode(ListNode node) {
        if (node.next.next == null) {
            node.val = node.next.val;
            node.next = null;
            return;
        }
        node.val = node.next.val;
        node.next = node.next.next;
    }


    /**
     * 238. Product of Array Except Self
     *
     * @param nums
     * @return
     */
    public int[] productExceptSelf(int[] nums) {
        if (nums == null || nums.length == 0) {
            return nums;
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


    /**
     * todo
     *
     * @param num
     * @return
     */
    public String numberToWords(int num) {
        if (num < 0) {
            return "";
        }

        return "";
    }

    private String convertToWords(int num) {
        String[] v1 = new String[]{"", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"};

        String[] v2 = new String[]{"", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"};


        int a = num / 100;

        int b = num % 100;

        int c = num % 10;

        String res = "";

//        res = b < 20 ? v1[b] : (v2[b / 10] +

        return res;

    }


    /**
     * 283. Move Zeroes
     *
     * @param nums
     */
    public void moveZeroes(int[] nums) {
        if (nums == null || nums.length == 0) {
            return;
        }
        int index = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != 0) {
                nums[index++] = nums[i];
            }
        }
        while (index < nums.length) {
            nums[index++] = 0;
        }
    }

    /**
     * 290. Word Pattern
     *
     * @param pattern
     * @param s
     * @return
     */
    public boolean wordPattern(String pattern, String s) {
        if (pattern == null || s == null) {
            return false;
        }
        String[] words = s.split(" ");
        if (pattern.length() != words.length) {
            return false;
        }
        Map<String, Integer> map2 = new HashMap<>();
        Map<Character, Integer> map1 = new HashMap<>();
        for (int i = 0; i < words.length; i++) {
            if (!Objects.equals(map1.put(pattern.charAt(i), i), map2.put(words[i], i))) {
                return false;
            }
        }
        return true;
    }


    /**
     * 242. Valid Anagram
     *
     * @param s
     * @param t
     * @return
     */
    public boolean isAnagram(String s, String t) {
        if (s == null || t == null) {
            return false;
        }
        if (s.equals(t)) {
            return false;
        }
        if (s.length() != t.length()) {
            return false;
        }
        int len = s.length();
        Map<Character, Integer> map1 = new HashMap<>();
        Map<Character, Integer> map2 = new HashMap<>();
        for (int i = 0; i < len; i++) {
            char s1 = s.charAt(i);
            char t1 = t.charAt(i);

            Integer count1 = map1.getOrDefault(s1, 0);
            Integer count2 = map2.getOrDefault(t1, 0);
            count2++;
            count1++;

            map1.put(s1, count1);
            map2.put(t1, count2);
        }
        for (int i = 0; i < len; i++) {
            if (!map1.get(s.charAt(i)).equals(map2.get(s.charAt(i)))) {
                return false;
            }
        }
        return true;
    }


    /**
     * todo
     * 299. Bulls and Cows
     *
     * @param secret
     * @param guess
     * @return
     */
    public String getHint(String secret, String guess) {
        int bulls = 0;
        int cows = 0;
        char[] secretWords = secret.toCharArray();
        char[] guessWords = guess.toCharArray();
        int[] hash = new int[10];
        for (int i = 0; i < secretWords.length; i++) {
            char secretWord = secretWords[i];
            char guessWord = guessWords[i];
            if (secretWord == guessWord) {
                bulls++;
            } else {
                if (hash[Character.getNumericValue(secretWord)]-- > 0) {
                    cows++;
                }
                if (hash[Character.getNumericValue(guessWord)]++ < 0) {
                    cows++;
                }
            }
        }
        return bulls + "A" + cows + "B";
    }

}
