package utils;

import java.util.HashSet;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * @author Akasaka Isami
 * @description 日志语句相关的工具类
 * <a href="https://regexr-cn.com/">参考这个网址</>
 * @date 2022-12-12 20:43:38
 */
public class LogUtil {

    private static final String regexGLOBAL = ".*?(log|trace|(system\\.out)|(system\\.err)).*?(.*?)";

    private static final Set<String> level_prix = new HashSet<String>() {{
        add("log");
        add("logger");
        add("logging");
        add("getlogger");
        add("getlog");
    }};

    private static final Set<String> levels = new HashSet<String>() {{
        add("trace");
        add("debug");
        add("info");
        add("warn");
        add("error");
        add("fatal");
    }};

    public static boolean isLogStatement(String statement, int flag) {
        String myRegex = ".*?(log|logger|logging|getlogger\\(\\)|getlog\\(\\))\\.(trace|debug|info|warn|error|fatal)\\(.*?\\)";
        Pattern p = Pattern.compile(myRegex, Pattern.CASE_INSENSITIVE);
        Matcher m = p.matcher(statement);
        return m.find();
    }


    /**
     * 正则表达式判断输入的语句是否是日志语句
     *
     * @param curStatement
     * @return
     */
    public static boolean isLogStatement(String curStatement) {
        //匹配引号中的内容
        Pattern p = Pattern.compile("\".*\"");
        Matcher m = p.matcher(curStatement);

        if (curStatement.toLowerCase().contains("assertequals") || curStatement.toLowerCase().contains("assertfalse") || curStatement.toLowerCase().contains("asserttrue")) {
            return false;
        }

        /* if find quotes */
        if (m.find()) {
            // 把引号中的东西删掉
            curStatement = curStatement.replaceAll("\".*?\"", "");

            // 判断是不是"log"相关的语句 比如说包含了诸如login、dialog这种关键字而被误判的语句
            if (!isLogRelated(curStatement))
                return false;

            // 找等号 找到等号说明这个语句应该是logger实例的赋值语句
            p = Pattern.compile("[^\"]*?\\=");
            Matcher mEqualSign = p.matcher(curStatement);

            if (mEqualSign.find())
                return false;

            // 日志语句的正则匹配
            p = Pattern.compile("(system\\.out)|(system.err)|(log(ger)?(\\(\\))?\\.(\\w*?)\\()|logauditevent\\(", Pattern.CASE_INSENSITIVE);
            m = p.matcher(curStatement);
            return m.find();

        } else {
            // 没引号就直接找等号 有等号的反正都不是日志语句
            p = Pattern.compile("[^\"]*?\\=");
            Matcher mEqualSign = p.matcher(curStatement);

            if (mEqualSign.find())
                return false;

            p = Pattern.compile("(system\\.out)|(system.err)|(log(ger)?(\\(\\))?\\.(\\w*?)\\()", Pattern.CASE_INSENSITIVE);
            m = p.matcher(curStatement);
            return m.find();
        }
    }


    /**
     * @param statement
     * @return
     */
    private static boolean isLogRelated(String statement) {
        //换行符？这个分支应该不会用
        if (statement.split("\r|\n|\r\n").length >= 3) // set as 3
        {
            return false;
        } else {

            Pattern p = Pattern.compile(regexGLOBAL, Pattern.CASE_INSENSITIVE | Pattern.DOTALL);
            Matcher m = p.matcher(statement);

            if (m.find()) {
                if (statement.toLowerCase().contains("system.out") || statement.toLowerCase().contains("system.err")) {
                    return true;
                }
                statement = statement.replaceAll("\"(.*?)\"", "");
                Pattern pKeyword = Pattern.compile("(login)|(dialog)|(logout)|(catalog)|logic(al)?", Pattern.CASE_INSENSITIVE);
                Pattern pTrueKeyword = Pattern.compile("loginput|logoutput", Pattern.CASE_INSENSITIVE);
                m = pKeyword.matcher(statement);
                Matcher m2 = pTrueKeyword.matcher(statement);
                return !m.find() || m2.find();
            }
        }
        return false;
    }

    public static void main(String[] args) {
        System.out.println("hi");
    }

}
