package utils;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class DotPrintFilter {
    private static final Pattern p;
    private static final Pattern dotP;
    private static final Pattern quotationP;

    private static final Pattern parenthesesP;
    private static final Pattern equalP;

    private static final Pattern allLowerP;
    private static final Pattern allUpperP;
    private static final Pattern numP;
    private static final Pattern capitalP;


    static {
        p = Pattern.compile("\r|\n|\r\n");
        dotP = Pattern.compile(",");
        quotationP = Pattern.compile("\"");

        parenthesesP = Pattern.compile("(.*)\\((.*)\\)"); // xxx(xxx)
        equalP = Pattern.compile("(.*)=('.*')"); // xxx='xxx'

        allLowerP = Pattern.compile("[a-z]+");
        allUpperP = Pattern.compile("[A-Z]+");
        numP = Pattern.compile("[0-9]+");
        capitalP = Pattern.compile("[A-Z][a-z]+");
    }

    public static String filterQuotation(String originalStr) {
        return originalStr.replaceAll("\"", "'");
    }

    public static String AstNodeFilter(String originalStr) {
        // 换行符拼接
        // 逗号换点号
        // 双引号换单引号
        Matcher matcher = p.matcher(originalStr);
        Matcher matcher2 = dotP.matcher(matcher.replaceAll(""));
        Matcher matcher3 = quotationP.matcher(matcher2.replaceAll("."));
        return matcher3.replaceAll("'");
    }

    public static String cut(String originalStr) {

        // ast节点的三种形态：
        // 1. xxx
        // 2. xxx(xxx)
        // 3. xxx='xxx'
        // 其中，xxx都可能是驼峰表示，或者是xxx_xxx的形势，所以要切分
        StringBuilder result = new StringBuilder();

        Matcher m1 = parenthesesP.matcher(originalStr);
        Matcher m2 = equalP.matcher(originalStr);

        try {
            if (m1.matches()) {
                // xxx(xxx)
                String[] ss = originalStr.split("\\(");
                String s1 = ss[0];
                String s2 = ss[1];
                s1 = s1.substring(0, s1.length() - 1);
                s2 = s2.substring(0, s2.length() - 1);
                s1 = cutHump(s1);
                s2 = cutHump(s2);

                result.append(s1).append(' ').append(s2);
                return result.toString();

            } else if (m2.matches()) {
                // xxx='xxx'
                String[] ss = originalStr.split("=", 2);
                String s1 = ss[0];
                String s2 = ss[1];
                s2 = s2.substring(1, s2.length() - 1);
                s1 = cutHump(s1);
                s2 = cutHump(s2);

                result.append(s1).append(' ').append('=').append(' ').append(s2);
            } else {
                result.append(cutHump(originalStr));
                return result.toString();
            }
        } catch (StringIndexOutOfBoundsException e) {
            throw new StringIndexOutOfBoundsException(originalStr + "出了问题");
        }


        return result.toString();
    }

    /**
     * 要处理的字符串包括
     * case1 AST 全小写 xxx e.g.variables
     * 直接返回
     * case2 AST hump XxxXxxXxx e.g. VariableDeclarationExpr
     * 切割转小写后拼接返回
     * case3 AST 全大写 XXX e.g. INT
     * 转小写后返回
     * case4 AST 全大写+下划线 XXX_XXX e.g. POSTFIX_INCREMENT
     * 切割转小写后拼接返回
     * ---------------------------------------------
     * case5 用户可能的变量定义 下划线 xxx_xxx e.g. akasaka_isami
     * 切割后返回
     * case6 用户可能的变量定义 数字 xxx12 e.g. str1、str2、a1、a2
     * 把数字和字母切开吧 = =
     * case7 用户可能的变量定义 大小写组合 e.g. getID IOException getASTCreater
     * 连续的大写被认为是一个单词 最后一个大写和后面的小写一起被认为是一个单词
     *
     * @param str 要处理的ast节点的字符串
     * @return 处理完以后切割好的字符串序列 用空格拼接
     */
    private static String cutHump(String str) throws StringIndexOutOfBoundsException {
        StringBuilder result = new StringBuilder();

        Matcher lowerMatcher = allLowerP.matcher(str);
        Matcher upperMatcher = allUpperP.matcher(str);
        Matcher numMatcher = numP.matcher(str);
        Matcher capitalMatcher = capitalP.matcher(str);
        if (str.contains(" ")) {
            if (str.trim().length() == 0)
                return str;
            else {
                String[] words = str.split(" ");
                for (String word : words) {
                    result.append(cutHump(word)).append(' ');
                }
                //去掉最后一个空格
                result.deleteCharAt(result.length() - 1);
            }
        } else if (str.contains("_")) {
            // 包含下划线
            String[] words = str.split("_");
            for (String word : words) {
                result.append(cutHump(word)).append(' ');
            }
            //去掉最后一个空格
            if (result.length() != 0)
                result.deleteCharAt(result.length() - 1);
        } else if (lowerMatcher.matches() || numMatcher.matches()) {
            // 全小写或全数字
            return str;
        } else if (upperMatcher.matches() || capitalMatcher.matches()) {
            // 全大写或首字母大写
            return str.toLowerCase();
        } else {
            // 有数字 有hump 乱七八糟的情况
            // 按照数字和大写字母分割吧 = =
            char[] chars = str.toCharArray();
            int n = chars.length;
            boolean isUpper = false;
            for (int i = 0; i < n; i++) {
                char c = chars[i];
                if (Character.isUpperCase(c)) {
                    // 大写 情况比较特殊
                    // 连续的大写被认为是一个单词 最后的大写和后面的小写一起被认为是一个单词
                    // 如果没前 那就直接加 +c
                    // 如果前面不是大写 +' '+c
                    // 如果前面是个大写
                    //      后面是个大写 +c
                    //      后面是个小写 +' '+c
                    //      后面是个数字 +c
                    //      没有后面 +c
                    if (result.length() == 0) {
                        result.append(Character.toLowerCase(c));
                    } else if (!isUpper) {
                        result.append(' ').append(Character.toLowerCase(c));
                    } else {
                        if (i == n - 1) {
                            result.append(Character.toLowerCase(c));
                        } else {
                            char next = chars[i + 1];
                            if (Character.isLowerCase(next)) {
                                result.append(' ').append(Character.toLowerCase(c));
                            } else result.append(Character.toLowerCase(c));
                        }

                    }
                    isUpper = true;
                } else if (Character.isLowerCase(c)) {
                    // 小写
                    // 如果前面是数字的话 先加空格；不是的话直接加
                    if (result.length() != 0 && Character.isDigit(result.charAt(result.length() - 1))) {
                        result.append(' ').append(c);
                    } else result.append(c);
                    isUpper = false;
                } else if (Character.isDigit(c)) {
                    // 数字
                    // 如果前面是数字的话 直接加；不是的话先加空格再加
                    if (result.length() != 0 && Character.isDigit(result.charAt(result.length() - 1))) {
                        result.append(c);
                    } else result.append(' ').append(c);
                    isUpper = false;
                }
            }

        }

        return result.toString();
    }

    public static void main(String[] args) {
        DotPrintFilter.cut("identifier='__'");
    }
}
