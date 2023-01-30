import com.github.javaparser.JavaParser;
import com.github.javaparser.ParseProblemException;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.visitor.VoidVisitor;
import config.MConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import visitor.MethodVisitor;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Objects;

/**
 * @author Akasaka Isami
 * @description 代码处理的入口 take the directory of the project as input
 * @date 2022-12-13 17:10:28
 */
public class Entry {
    private static final Logger logger = LoggerFactory.getLogger(Entry.class);


    public static void main(String[] args) throws FileNotFoundException {
//        File test = new File(MConfig.rootDir + MConfig.primaryDir + MConfig.projectName);
//        if (!test.exists())
//            FileUtil.moveFiles(MConfig.rootDir + MConfig.srcDir, MConfig.rootDir + MConfig.primaryDir, MConfig.projectName);

        String srcDirPath = MConfig.rootDir + MConfig.projectName + '/' + MConfig.primaryDir;
        File srcDir = new File(srcDirPath);
        if (!srcDir.isDirectory()) {
            return;
        }

        List<String> parseFailFiles = new ArrayList<>();
        int count = 0;
        long startTime = new Date().getTime();

        // 遍历所有文件
        for (File file : Objects.requireNonNull(srcDir.listFiles())) {
            String fileName = file.getName();

            logger.info("Entry: 正在解析文件" + fileName);


            try {
                CompilationUnit cu = JavaParser.parse(file);

                VoidVisitor<String> methodVisitor = new MethodVisitor();
                methodVisitor.visit(cu, fileName);
                count++;
            } catch (ParseProblemException e) {
                parseFailFiles.add(fileName);
                e.printStackTrace();
                logger.warn(fileName + "解析出错，直接跳过");
            }
        }
        long endTime = new Date().getTime();

        logger.info("数据集构建完成，成功解析" + count + "个文件");
        if (!parseFailFiles.isEmpty()) {
            logger.info("解析错误的文件有" + parseFailFiles.size() + "个");
            StringBuilder allFailFiles = new StringBuilder();

            for (String failFile : parseFailFiles) {
                allFailFiles.append(failFile).append("、");
            }

            logger.info("解析错误的文件:" + allFailFiles);
        }

        logger.info("运行时间： " + (endTime - startTime) / 1000 + "秒");
        logger.info("总共日志函数数量：" + MethodVisitor.totalLoggedMethod);


    }


}
