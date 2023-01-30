package view;

import config.MConfig;
import model.AstNode;
import model.EdgeTypes;
import model.GraphEdge;
import model.GraphNode;
import utils.DotPrintFilter;
import utils.LogUtil;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.*;


public class CFG_ASTPrinter2 {

    private String path;
    private StringBuilder str;
    private List<String> sentences;

    private int index2 = 0;
    private List<String> leafNodes;

    // key: 语句的节点编号  value：语句的ast dot字符串
    // 遍历这个map，写入语句ast文件
    private Map<String, String> ASTStrMap;

    private Set<GraphEdge> allDFGEdgesList;


    public CFG_ASTPrinter2(String path, Set<GraphEdge> allDFGEdgesList) {
        this.path = path;
        str = new StringBuilder("digraph {");
        leafNodes = new ArrayList<>();
        ASTStrMap = new HashMap<>();
        this.allDFGEdgesList = allDFGEdgesList;
        this.sentences = new ArrayList<>();
    }

    private void BFS(GraphNode root) {
        Queue<GraphNode> dealingNodes = new LinkedList<>();
        dealingNodes.add(root);

        Set<String> visited = new HashSet<>();
        while (!dealingNodes.isEmpty()) {
            GraphNode par = dealingNodes.poll();
            if (visited.contains(par.getDotNum())) {
            } else {
                boolean isLogStmt = LogUtil.isLogStatement(par.getOriginalCodeStr(), 1);

                if (isLogStmt) {
                    String level = "";
                    // 如果是日志语句，要把name->identifier里的内容改成mask
                    for (AstNode tempNode : par.getAstRootNode().getSubNodes()) {
                        if (tempNode.getTypeName().equals("name (SimpleName)")) {
                            String tempStr = tempNode.getAttributes().get(0);
                            level = tempStr.substring(12, tempStr.length() - 1);
                            tempNode.getAttributes().set(0, "identifier='mask'");
                        }
                    }
                    str.append(System.lineSeparator()).append(par.getDotNum()).append(" [label=\"").append(DotPrintFilter.filterQuotation(par.getOriginalCodeStr())).append("\" , line=").append(par.getCodeLineNum()).append(", isLogStmt=\"").append(isLogStmt).append("\", level=" + level + "];");
                } else {
                    str.append(System.lineSeparator()).append(par.getDotNum()).append(" [label=\"").append(DotPrintFilter.filterQuotation(par.getOriginalCodeStr())).append("\" , line=").append(par.getCodeLineNum()).append(", isLogStmt=\"").append(isLogStmt).append("\"];");
                }


                // 我改的 创建AST的代码
                String dotnum = par.getDotNum();
                StringBuilder value = new StringBuilder();
                value.append("digraph {");

                StringBuilder sentence = new StringBuilder();
                // 这里分两种情况 第一种情况是这个node有ast树 那就递归写入ast文件就好了
                // 第二种情况是 这个node没有ast树 那就创建一个单节点的树 把当前节点的值写进去就行了
                // 一般没有ast的都是简单节点 比如"return" "case"之类的
                if (par.getAstRootNode() == null) {
                    String node_value = par.getOpTypeStr();
                    String ndName = "n" + index2;
                    value.append(System.lineSeparator()).append(ndName).append(" [label=\"").append(node_value).append("\"];");

                    sentence.append(DotPrintFilter.cut(node_value));
                    sentences.add(sentence.toString());
                } else {
                    dfs(par.getAstRootNode(), "", value, sentence);

                    // 去掉语料库最后的空格
                    if (sentence.length() != 0) sentence.deleteCharAt(sentence.length() - 1);
                    sentences.add(sentence.toString());
                }


                value.append(System.lineSeparator()).append("}");
                index2 = 0;
                ASTStrMap.put(dotnum, value.toString());


                visited.add(par.getDotNum());

            }


            // 遍历邻接节点
            List<GraphNode> adjacentPoints = par.getAdjacentPoints();
            for (GraphNode child : adjacentPoints) {
                if (visited.contains(child.getDotNum())) {
                } else {
                    dealingNodes.add(child);
                    boolean isLogStmt = LogUtil.isLogStatement(child.getOriginalCodeStr(), 1);
                    if (isLogStmt) {
                        String level = "";
                        // 如果是日志语句，要把name->identifier里的内容改成mask
                        for (AstNode tempNode : child.getAstRootNode().getSubNodes()) {
                            if (tempNode.getTypeName().equals("name (SimpleName)")) {
                                String tempStr = tempNode.getAttributes().get(0);
                                level = tempStr.substring(12, tempStr.length() - 1);
                                tempNode.getAttributes().set(0, "identifier='mask'");
                            }
                        }
                        str.append(System.lineSeparator()).append(child.getDotNum()).append(" [label=\"").append(DotPrintFilter.filterQuotation(child.getOriginalCodeStr())).append("\" , line=").append(child.getCodeLineNum()).append(", isLogStmt=\"").append(isLogStmt).append("\", level=" + level + "];");
                    } else {
                        str.append(System.lineSeparator()).append(child.getDotNum()).append(" [label=\"").append(DotPrintFilter.filterQuotation(child.getOriginalCodeStr())).append("\" , line=").append(child.getCodeLineNum()).append(", isLogStmt=\"").append(isLogStmt).append("\"];");
                    }

                    // 我改的 创建AST的代码
                    String dotnum = child.getDotNum();
                    StringBuilder value = new StringBuilder();
                    value.append("digraph {");

                    StringBuilder sentence = new StringBuilder();
                    // 这里分两种情况 第一种情况是这个node有ast树 那就递归写入ast文件就好了
                    // 第二种情况是 这个node没有ast树 那就创建一个单节点的树 把当前节点的值写进去就行了
                    // 一般没有ast的都是简单节点 比如"return" "break" "finally" "else"之类的

                    if (child.getAstRootNode() == null) {
                        String node_value = child.getOpTypeStr();
                        String ndName = "n" + index2;
                        value.append(System.lineSeparator()).append(ndName).append(" [label=\"").append(node_value).append("\"];");

                        sentence.append(DotPrintFilter.cut(node_value));
                        sentences.add(sentence.toString());
                    } else {
                        dfs(child.getAstRootNode(), "", value, sentence);

                        // 去掉语料库最后的空格
                        if (sentence.length() != 0) sentence.deleteCharAt(sentence.length() - 1);
                        sentences.add(sentence.toString());
                    }


                    value.append(System.lineSeparator()).append("}");
                    index2 = 0;
                    ASTStrMap.put(dotnum, value.toString());


                    visited.add(child.getDotNum());
                }
            }


            for (GraphEdge edge : par.getEdgs()) {
                str.append(System.lineSeparator()).append(edge.getOriginalNode().getDotNum()).append(" -> ").append(edge.getAimNode().getDotNum()).append("[color=").append(edge.getType().getColor()).append("];");
            }


        }

        for (GraphEdge edge : this.allDFGEdgesList) {
            str.append(System.lineSeparator() + edge.getOriginalNode().getDotNum() + " -> " + edge.getAimNode().getDotNum() + "[color=" + edge.getType().getColor() + "];");
        }


        str.append(System.lineSeparator() + "}");

    }


    private void dfs(AstNode node, String parentNodeName, StringBuilder str, StringBuilder sentence) {
        if (node != null) {

            List<String> attributes = node.getAttributes();
            List<AstNode> subNodes = node.getSubNodes();
            List<String> subLists = node.getSubLists();
            List<List<AstNode>> subListNodes = node.getSubListNodes();

            String ndName = "n" + (index2++);

            if (!node.toString().equals("")) {
                String label = DotPrintFilter.AstNodeFilter(node.getTypeName());
                sentence.append(DotPrintFilter.cut(label)).append(' ');
                str.append(System.lineSeparator()).append(ndName).append(" [label=\"").append(DotPrintFilter.AstNodeFilter(node.getTypeName())).append("\", ast_node=\"true\"];");
            }

            if (!parentNodeName.isEmpty()) {
                str.append(System.lineSeparator()).append(parentNodeName).append(" -> ").append(ndName).append("[color=").append(EdgeTypes.AST.getColor()).append("];");
            }

            for (String a : attributes) {
                String label = DotPrintFilter.AstNodeFilter(a);
                sentence.append(DotPrintFilter.cut(label)).append(' ');
                String attrName = "n" + (index2++);
                str.append(System.lineSeparator()).append(attrName).append(" [label=\"").append(DotPrintFilter.AstNodeFilter(a)).append("\", ast_node=\"true\"];");
                str.append(System.lineSeparator()).append(ndName).append(" -> ").append(attrName).append("[color=").append(EdgeTypes.AST.getColor()).append("];");
            }

            for (AstNode subNode : subNodes) {
                dfs(subNode, ndName, str, sentence);
            }

            for (int i = 0; i < subLists.size(); i++) {
                String label = DotPrintFilter.AstNodeFilter(subLists.get(i));
                sentence.append(DotPrintFilter.cut(label)).append(' ');

                String ndLstName = "n" + (index2++);
                str.append(System.lineSeparator()).append(ndLstName).append(" [label=\"").append(DotPrintFilter.AstNodeFilter(subLists.get(i))).append("\", ast_node=\"true\"];");
                str.append(System.lineSeparator()).append(ndName).append(" -> ").append(ndLstName).append("[color=").append(EdgeTypes.AST.getColor()).append("];");

                for (int j = 0; j < subListNodes.get(i).size(); j++) {
                    dfs(subListNodes.get(i).get(j), ndLstName, str, sentence);
                }

            }


        }
    }


    public void print(GraphNode root, String methodName, String uniqueMethodName, boolean ncs) {
        BFS(root);

        File file = new File(path);
        if (!file.exists()) {
            file.mkdirs();
        }
        try {
            BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(
                    new File(path + "/" + uniqueMethodName + "_CA.dot")));
            bufferedWriter.write(str.toString());
            bufferedWriter.flush();
            bufferedWriter.close();
        } catch (Exception e) {
            System.out.println("数据写入ast文件发送异常！");
        }


        // 我写的
        try {
            File ASTfile = new File(path + "/statements@" + uniqueMethodName + ".dot");
            BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(ASTfile));


            List<List<String>> ASTStrList = new ArrayList<>();


            for (Map.Entry<String, String> entry : ASTStrMap.entrySet()) {
                List<String> temp = new ArrayList<>();
                temp.add(entry.getKey());
                temp.add(entry.getValue());

                ASTStrList.add(temp);
            }

            // 根据ASTStrList写个自定义排序
            Collections.sort(ASTStrList, (o1, o2) -> {
                String key1 = o1.get(0);
                int n1 = Integer.parseInt(key1.substring(1));
                String key2 = o2.get(0);
                int n2 = Integer.parseInt(key2.substring(1));
                return n1 - n2;
            });

            for (List<String> pair :
                    ASTStrList) {
                bufferedWriter.write(pair.get(1));
                bufferedWriter.write("\n");
            }


            bufferedWriter.flush();
            bufferedWriter.close();

        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("数据写入ast文件发送异常！");
        }


        try (FileWriter writer = new FileWriter(MConfig.rootDir + MConfig.projectName + "/" + MConfig.rawDir + MConfig.projectName + "_corpus.txt", true);
             BufferedWriter bw = new BufferedWriter(writer)) {
            for (String sentence : sentences) {
                bw.append(sentence);
                bw.newLine();
            }
        } catch (Exception e) {
            System.out.println("数据写入语料库文件异常！");
        }


    }

    private GraphNode findNotLogAncestor(GraphNode node) {
        while (node != null) {
            if (!LogUtil.isLogStatement(node.getOriginalCodeStr(), 1)) {
                return node;
            }

            if (!node.getPreAdjacentPoints().isEmpty()) {
                node = node.getPreAdjacentPoints().get(0);
            } else node = node.getParentNode();
        }
        return node;
    }
}
