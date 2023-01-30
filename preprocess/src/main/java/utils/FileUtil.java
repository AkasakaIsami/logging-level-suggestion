package utils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.File;

/**
 * @author Akasaka Isami
 * @description 文件IO相关的工具类
 * @date 2022-11-22 20:53:38
 */
public class FileUtil {

    private static final Logger logger = LoggerFactory.getLogger(FileUtil.class);


    /**
     * 将目标文件下的所有java文件移动到指定文件，并重命名文件为”项目名+文件名“
     * 路径皆默认绝对路径
     *
     * @param srcDirPath    源文件目录
     * @param targetDirPath 目标目录
     * @param pjName        项目名
     */
    public static void moveFiles(String srcDirPath, String targetDirPath, String pjName) {
        File srcDir = new File(srcDirPath + pjName);
        File targetDir = new File(targetDirPath + pjName);

        if (!srcDir.exists()) {
            logger.info("源数据文件夹不存在");
            return;
        }

        if (!targetDir.exists()) {
            targetDir.mkdirs();
        }

        if (!srcDir.isDirectory() || !targetDir.isDirectory()) {
            logger.info("源数据文件或目标文件非文件夹");
            return;
        }

        recMoveFiles(srcDir, targetDir, pjName);
        logger.info("moveFiles：完成移动并重命名文件");

    }


    /**
     * 返回指定文件内的所有内容
     *
     * @param path 文件路径
     * @return 返回文件中所有内容的字符流
     * @throws Exception
     */
    public static String getFileString(String path) throws Exception {
        BufferedReader reader = new BufferedReader(new java.io.FileReader(path));
        StringBuilder result = new StringBuilder();
        String line = "";

        while ((line = reader.readLine()) != null) {
            result.append(line).append("\n");
        }

        reader.close();
        return result.toString();
    }


    private static void recMoveFiles(File root, File targetDir, String pjName) {
        File[] files = root.listFiles();
        assert files != null;
        for (File file : files) {
            if (file.isDirectory()) {
                recMoveFiles(file, targetDir, pjName);
            } else {
                if (isJavaFile(file.getName()))
                    moveFile(file, targetDir, pjName);
            }
        }
    }

    private static void moveFile(File file, File targetDir, String pjName) {
        String newName = targetDir + "/" + pjName + '_' + file.getName();
        file.renameTo(new File(newName));

    }

    private static boolean isJavaFile(String filename) {
        int index = filename.indexOf(".");

        if (index == -1) {
            return false;
        }
        return filename.substring(index + 1).equals("java");
    }

    public static String extractFileName(String filename) {
        if (isJavaFile(filename)) {
            return filename.substring(0, filename.indexOf("."));
        }
        return null;
    }

}
