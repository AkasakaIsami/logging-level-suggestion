package service;


import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.NodeList;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.expr.Expression;
import com.github.javaparser.ast.stmt.*;
import model.AstNode;
import model.GraphNode;
import org.apache.commons.lang3.StringUtils;
import utils.LogUtil;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Optional;


public class ASTCreater {

    private HashMap<String, GraphNode> allCFGNodesMap;

    public ASTCreater(HashMap<String, GraphNode> allCFGNodesMap) {
        this.allCFGNodesMap = allCFGNodesMap;
    }

    public void buildMethodAST(Node node) {
        if (node instanceof MethodDeclaration) {
            MethodDeclaration methodDeclaration = ((MethodDeclaration) node).asMethodDeclaration();
//            System.out.println("********************************************");
//            System.out.println("当前正在生成AST节点方法的名字：" + methodDeclaration.getDeclarationAsString(false, false, true));
//            System.out.println("********************************************");
            String label = methodDeclaration.getDeclarationAsString(false, true, true);
            int lineNum = methodDeclaration.getBegin().isPresent() ? methodDeclaration.getBegin().get().line : -1;

            GraphNode methodNode = allCFGNodesMap.get(label + ":" + lineNum);


            AstNode astNode = new AstNode();
            AstNodeInit astNodeInit = new AstNodeInit(true, astNode);
            astNodeInit.Init(methodDeclaration);
            methodNode.setAstRootNode(astNode);

            Optional<BlockStmt> body = methodDeclaration.getBody();
            if (body.isPresent()) {
                NodeList<Statement> statements = body.get().getStatements();
                for (Statement statement : statements) {
                    buildAST(statement);
                }
            }
        }
    }

    private void buildAST(Node node) {
        if (node instanceof ExpressionStmt) {
            ExpressionStmt exStmt = ((ExpressionStmt) node).asExpressionStmt();
            Expression expression = exStmt.getExpression();
            String label = expression.toString();
            int lineNum = expression.getBegin().isPresent() ? expression.getBegin().get().line : -1;
            GraphNode cfgNode = allCFGNodesMap.get(label + ":" + lineNum);
            if (cfgNode == null) {
                System.out.println("stop");
            }

            AstNode astNode = new AstNode();
            AstNodeInit astNodeInit = new AstNodeInit(true, astNode);
            astNodeInit.Init(expression);

            assert cfgNode != null;
            boolean isLogStmt = LogUtil.isLogStatement(cfgNode.getOriginalCodeStr(), 1);
            
            cfgNode.setAstRootNode(astNode);
        } else if (node instanceof IfStmt) {
            IfStmt tempIfStmt = ((IfStmt) node).asIfStmt();
            while (tempIfStmt != null) {
                String ifLabel = "if (" + tempIfStmt.getCondition().toString() + ")";
                int ifLineNum = tempIfStmt.getBegin().isPresent() ? tempIfStmt.getBegin().get().line : -1;
                GraphNode ifCfgNode = allCFGNodesMap.get(ifLabel + ":" + ifLineNum);
                AstNode ifAstNode = new AstNode();
                AstNodeInit astNodeInit = new AstNodeInit(true, ifAstNode);
//                astNodeInit.Init(tempIfStmt.getCondition());
                astNodeInit.Init(tempIfStmt);
                ifCfgNode.setAstRootNode(ifAstNode);
                if (!tempIfStmt.getThenStmt().isBlockStmt()) {
                    buildAST(tempIfStmt.getThenStmt());
                } else {
                    BlockStmt thenBlockStmt = tempIfStmt.getThenStmt().asBlockStmt();
                    NodeList<Statement> statements = thenBlockStmt.getStatements();
                    for (Statement statement : statements) {
                        buildAST(statement);
                    }
                }
                if (tempIfStmt.getElseStmt().isPresent()) {
                    if (tempIfStmt.getElseStmt().get().isIfStmt()) {
                        tempIfStmt = tempIfStmt.getElseStmt().get().asIfStmt();
                    } else {
                        if (!tempIfStmt.getElseStmt().get().isBlockStmt()) {
                            buildAST(tempIfStmt.getElseStmt().get());
                        } else {
                            BlockStmt elseBlockStmt = tempIfStmt.getElseStmt().get().asBlockStmt();
                            NodeList<Statement> statements1 = elseBlockStmt.getStatements();
                            for (Statement statement : statements1) {
                                buildAST(statement);
                            }
                        }
                        tempIfStmt = null;
                    }
                } else {
                    tempIfStmt = null;
                }
            }
        } else if (node instanceof WhileStmt) {
            WhileStmt whileStmt = ((WhileStmt) node).asWhileStmt();
            String label = "while (" + whileStmt.getCondition().toString() + ")";
            int lineNum = whileStmt.getBegin().isPresent() ? whileStmt.getBegin().get().line : -1;
            GraphNode cfgNode = allCFGNodesMap.get(label + ":" + lineNum);
            AstNode astNode = new AstNode();
            AstNodeInit astNodeInit = new AstNodeInit(true, astNode);
            astNodeInit.Init(whileStmt);
            cfgNode.setAstRootNode(astNode);

            if (!whileStmt.getBody().isBlockStmt()) {
                buildAST(whileStmt.getBody());
            } else {
                NodeList<Statement> statements = whileStmt.getBody().asBlockStmt().getStatements();
                if (statements.size() == 0) {
                    return;
                }
                for (Statement statement : statements) {
                    buildAST(statement);
                }
            }
        } else if (node instanceof ForStmt) {
            List<String> forValues = new ArrayList<>();
            ForStmt forStmt = ((ForStmt) node).asForStmt();
            forValues.add(StringUtils.join(forStmt.getInitialization(), ","));
            if (forStmt.getCompare().isPresent()) {
                forValues.add(forStmt.getCompare().get().toString());
            }
            forValues.add(StringUtils.join(forStmt.getUpdate(), ","));
            String label = "for(" + StringUtils.join(forValues, ';') + ")";
            int lineNum = forStmt.getBegin().isPresent() ? forStmt.getBegin().get().line : -1;
            GraphNode cfgNode = allCFGNodesMap.get(label + ":" + lineNum);
            AstNode astNode = new AstNode();
            AstNodeInit astNodeInit = new AstNodeInit(true, astNode);
            astNodeInit.Init(forStmt);
            cfgNode.setAstRootNode(astNode);

            if (!forStmt.getBody().isBlockStmt()) {
                buildAST(forStmt.getBody());
            } else {
                NodeList<Statement> statements = forStmt.getBody().asBlockStmt().getStatements();
                if (statements.size() == 0) {
                    return;
                }
                for (Statement statement : statements) {
                    buildAST(statement);
                }
            }
        } else if (node instanceof ForeachStmt) {
            ForeachStmt foreachStmt = ((ForeachStmt) node).asForeachStmt();
            String label = "for(" + foreachStmt.getVariable() + ":" + foreachStmt.getIterable() + ")";
            int lineNum = foreachStmt.getBegin().isPresent() ? foreachStmt.getBegin().get().line : -1;
            GraphNode cfgNode = allCFGNodesMap.get(label + ":" + lineNum);
            AstNode astNode = new AstNode();
            AstNodeInit astNodeInit = new AstNodeInit(true, astNode);
            astNodeInit.Init(foreachStmt);
            cfgNode.setAstRootNode(astNode);

            if (!foreachStmt.getBody().isBlockStmt()) {
                buildAST(foreachStmt.getBody());
            } else {
                NodeList<Statement> statements = foreachStmt.getBody().asBlockStmt().getStatements();
                if (statements.size() == 0) {
                    return;
                }
                for (Statement statement : statements) {
                    buildAST(statement);
                }
            }
        } else if (node instanceof SwitchStmt) {
            SwitchStmt switchStmt = ((SwitchStmt) node).asSwitchStmt();

            String label = "switch(" + switchStmt.getSelector().toString() + ")";
            int lineNum = switchStmt.getBegin().isPresent() ? switchStmt.getBegin().get().line : -1;
            GraphNode cfgNode = allCFGNodesMap.get(label + ":" + lineNum);
            AstNode astNode = new AstNode();
            AstNodeInit astNodeInit = new AstNodeInit(true, astNode);
            astNodeInit.Init(switchStmt.getSelector());
            cfgNode.setAstRootNode(astNode);

            NodeList<SwitchEntryStmt> caseEntries = switchStmt.getEntries();
            if (caseEntries.size() == 0) {
                return;
            }
            for (int i = 0; i < caseEntries.size(); i++) {
                NodeList<Statement> statements = caseEntries.get(i).getStatements(); //一个case下面的所有语句
                label = caseEntries.get(i).getLabel().isPresent() ? "case " + caseEntries.get(i).getLabel().get().toString() : "default";
                lineNum = caseEntries.get(i).getBegin().isPresent() ? caseEntries.get(i).getBegin().get().line : -1;

                //case 增加ast node信息
                if (caseEntries.get(i).getLabel().isPresent()) {
                    GraphNode caseCfgNode = allCFGNodesMap.get(label + ":" + lineNum);
                    //处理ast
                    AstNode caseAstNode = new AstNode();
                    caseAstNode.setTypeName(caseEntries.get(i).getLabel().get().toString());
                    caseCfgNode.setAstRootNode(caseAstNode);
                }

                for (Statement statement : statements) {
                    buildAST(statement);
                }
            }

        } else if (node instanceof DoStmt) {
            DoStmt doStmt = ((DoStmt) node).asDoStmt();
            String label = "while (" + doStmt.getCondition().toString() + ")";
            int lineNum = doStmt.getCondition().getBegin().isPresent() ? doStmt.getCondition().getBegin().get().line : -1;
            GraphNode cfgNode = allCFGNodesMap.get(label + ":" + lineNum);
            AstNode astNode = new AstNode();
            AstNodeInit astNodeInit = new AstNodeInit(true, astNode);
//            astNodeInit.Init(doStmt.getCondition());
            astNodeInit.Init(doStmt);
            cfgNode.setAstRootNode(astNode);

            if (!doStmt.getBody().isBlockStmt()) {
                buildAST(doStmt.getBody());
            } else {
                NodeList<Statement> statements = doStmt.getBody().asBlockStmt().getStatements();
                for (Statement statement : statements) {
                    buildAST(statement);
                }
            }

        } else if (node instanceof BreakStmt) {
            BreakStmt breakStmt = ((BreakStmt) node).asBreakStmt();

            String label = breakStmt.getLabel().isPresent() ? "break " + breakStmt.getLabel().get().toString() : "break";
            int lineNum = breakStmt.getBegin().isPresent() ? breakStmt.getBegin().get().line : -1;

            //case 增加ast node信息
            if (breakStmt.getLabel().isPresent()) {
                GraphNode caseCfgNode = allCFGNodesMap.get(label + ":" + lineNum);
                //处理ast
                AstNode caseAstNode = new AstNode();
                caseAstNode.setTypeName(breakStmt.getLabel().get().toString());
                caseCfgNode.setAstRootNode(caseAstNode);
            }
        } else if (node instanceof ContinueStmt) {
            ContinueStmt continueStmt = ((ContinueStmt) node).asContinueStmt();

            String label = continueStmt.getLabel().isPresent() ? "continue " + continueStmt.getLabel().get().toString() : "continue";
            int lineNum = continueStmt.getBegin().isPresent() ? continueStmt.getBegin().get().line : -1;

            //case 增加ast node信息
            if (continueStmt.getLabel().isPresent()) {
                GraphNode caseCfgNode = allCFGNodesMap.get(label + ":" + lineNum);
                //处理ast
                AstNode caseAstNode = new AstNode();
                caseAstNode.setTypeName(continueStmt.getLabel().get().toString());
                caseCfgNode.setAstRootNode(caseAstNode);
            }
        } else if (node instanceof LabeledStmt) {
            LabeledStmt labeledStmt = ((LabeledStmt) node).asLabeledStmt();
            buildAST(labeledStmt.getStatement());
        } else if (node instanceof ReturnStmt) {
            ReturnStmt returnStmt = ((ReturnStmt) node).asReturnStmt();

            String label = returnStmt.getExpression().isPresent() ? "return " + returnStmt.getExpression().get().toString() : "return";
            int lineNum = returnStmt.getBegin().isPresent() ? returnStmt.getBegin().get().line : -1;
            if (returnStmt.getExpression().isPresent()) {
                GraphNode cfgNode = allCFGNodesMap.get(label + ":" + lineNum);
                //处理ast
                AstNode astNode = new AstNode();
                AstNodeInit astNodeInit = new AstNodeInit(true, astNode);
                astNodeInit.Init(returnStmt);
                cfgNode.setAstRootNode(astNode);
            }
        } else if (node instanceof AssertStmt) {
            AssertStmt assertStmt = ((AssertStmt) node).asAssertStmt();

            String label = assertStmt.getMessage().isPresent() ? "assert" + assertStmt.getCheck().toString() + ";" + assertStmt.getMessage().get().toString() : "assert" + assertStmt.getCheck().toString();
            int lineNum = assertStmt.getBegin().isPresent() ? assertStmt.getBegin().get().line : -1;
            GraphNode cfgNode = allCFGNodesMap.get(label + ":" + lineNum);
            //处理ast
            AstNode astNode = new AstNode();
            AstNodeInit astNodeInit = new AstNodeInit(true, astNode);
            astNodeInit.Init(assertStmt);
            cfgNode.setAstRootNode(astNode);
        } else if (node instanceof ThrowStmt) {
            ThrowStmt throwStmt = ((ThrowStmt) node).asThrowStmt();

            String label = "throw " + throwStmt.getExpression();
            int lineNum = throwStmt.getBegin().isPresent() ? throwStmt.getBegin().get().line : -1;
            GraphNode cfgNode = allCFGNodesMap.get(label + ":" + lineNum);
            //处理ast
            AstNode astNode = new AstNode();
            AstNodeInit astNodeInit = new AstNodeInit(true, astNode);
            astNodeInit.Init(throwStmt);
            cfgNode.setAstRootNode(astNode);
        } else if (node instanceof SynchronizedStmt) {
            SynchronizedStmt synchronizedStmt = ((SynchronizedStmt) node).asSynchronizedStmt();

            String label = "synchronized (" + synchronizedStmt.getExpression() + ")";
            int lineNum = synchronizedStmt.getBegin().isPresent() ? synchronizedStmt.getBegin().get().line : -1;
            GraphNode cfgNode = allCFGNodesMap.get(label + ":" + lineNum);
            //处理ast
            AstNode astNode = new AstNode();
            AstNodeInit astNodeInit = new AstNodeInit(true, astNode);
            astNodeInit.Init(synchronizedStmt);
            cfgNode.setAstRootNode(astNode);

            BlockStmt body = synchronizedStmt.getBody();
            NodeList<Statement> statements = body.getStatements();
            for (Statement statement : statements) {
                //开始递归创建，每处理一个statement返回一个node，这个node作为下一个state的前驱点
                buildAST(statement); //外面这个方法节点是方法体所有state的父节点
            }

        } else if (node instanceof BlockStmt) {
            BlockStmt blockStmt = ((BlockStmt) node).asBlockStmt();
            NodeList<Statement> statements = blockStmt.getStatements();
            for (Statement statement : statements) {
                //开始递归创建，每处理一个statement返回一个node，这个node作为下一个state的前驱点
                buildAST(statement);
            }
        } else if (node instanceof TryStmt) {
            TryStmt tryStmt = ((TryStmt) node).asTryStmt();
            String label = tryStmt.getResources().size() == 0 ? "try" : "try(" + StringUtils.join(tryStmt.getResources(), ";") + ")";
            int lineNum = tryStmt.getBegin().isPresent() ? tryStmt.getBegin().get().line : -1;
            GraphNode cfgNode = allCFGNodesMap.get(label + ":" + lineNum);
            //处理ast
            AstNode astNode = new AstNode();
            AstNodeInit astNodeInit = new AstNodeInit(true, astNode);
            astNodeInit.Init(tryStmt);
            cfgNode.setAstRootNode(astNode);

            BlockStmt tryBlock = tryStmt.getTryBlock();
            NodeList<Statement> statements = tryBlock.getStatements();
            for (Statement statement : statements) {
                buildAST(statement);
            }

            // 我写的
            NodeList<CatchClause> catchClauses = tryStmt.getCatchClauses();
            if (!catchClauses.isEmpty()) {
                for (CatchClause catchClause : catchClauses) {
                    String catchLabel = "catch (" + catchClause.getParameter().getType().toString() + " " + catchClause.getParameter().getName().toString() + ")";
                    int catchLineNum = catchClause.getBegin().isPresent() ? catchClause.getBegin().get().line : -1;
                    GraphNode catchNode = allCFGNodesMap.get(catchLabel + ":" + catchLineNum);

                    AstNode catchAstNode = new AstNode();
                    AstNodeInit catchAstNodeInit = new AstNodeInit(true, catchAstNode);
                    catchAstNodeInit.Init(catchClause);
                    catchNode.setAstRootNode(catchAstNode);

                    BlockStmt catchBody = catchClause.getBody();
                    NodeList<Statement> catchStatements = catchBody.getStatements();
                    for (Statement statement : catchStatements) {
                        buildAST(statement);
                    }
                }
            }

            Optional<BlockStmt> finallyBlock = tryStmt.getFinallyBlock();
            if (finallyBlock.isPresent()) {
                NodeList<Statement> finaBodyStas = finallyBlock.get().getStatements();
                for (Statement statement : finaBodyStas) {
                    buildAST(statement);
                }
            }
        } else if (node instanceof LocalClassDeclarationStmt) {
            LocalClassDeclarationStmt localClassDeclarationStmt = ((LocalClassDeclarationStmt) node).asLocalClassDeclarationStmt();
            ClassOrInterfaceDeclaration classOrInterfaceDeclaration = localClassDeclarationStmt.getClassDeclaration();

            String label = classOrInterfaceDeclaration.getNameAsString();
            int lineNum = classOrInterfaceDeclaration.getBegin().isPresent() ? classOrInterfaceDeclaration.getBegin().get().line : -1;

            GraphNode cfgNode = allCFGNodesMap.get(label + ":" + lineNum);
            //处理ast
            AstNode astNode = new AstNode();
            AstNodeInit astNodeInit = new AstNodeInit(true, astNode);
            astNodeInit.Init(classOrInterfaceDeclaration);
            cfgNode.setAstRootNode(astNode);
        }
    }
}
