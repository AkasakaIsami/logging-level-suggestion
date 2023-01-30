package service;

import com.github.javaparser.ast.ArrayCreationLevel;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.NodeList;
import com.github.javaparser.ast.body.*;
import com.github.javaparser.ast.expr.*;
import com.github.javaparser.ast.stmt.*;
import model.DFVarNode;
import model.EdgeTypes;
import model.GraphEdge;
import model.GraphNode;
import org.apache.commons.lang3.StringUtils;
import utils.LogUtil;

import java.util.*;

public class DFGCreater {

    private HashMap<String, GraphNode> allCFGNodesMap;

    private Set<GraphEdge> allDFGEdgesList;

    public DFGCreater(HashMap<String, GraphNode> allCFGNodesMap) {
        this.allCFGNodesMap = allCFGNodesMap;
        this.allDFGEdgesList = new HashSet<>();
    }


    public void buildMethodDFG(Node node) {
        if (node instanceof MethodDeclaration) {
            MethodDeclaration methodDeclaration = ((MethodDeclaration) node).asMethodDeclaration();
            if (methodDeclaration.getParentNode().isPresent()) {
                if (!(methodDeclaration.getParentNode().get() instanceof TypeDeclaration)) {
                    return;
                }
            }
//            System.out.println("********************************************");
//            System.out.println("当前正在CFG基础上生成DFG方法的名字：" + methodDeclaration.getDeclarationAsString(false, false, true));
//            System.out.println("********************************************");
            String label = methodDeclaration.getDeclarationAsString(false, true, true);
            int lineNum = methodDeclaration.getBegin().isPresent() ? methodDeclaration.getBegin().get().line : -1;
            GraphNode methodNode = allCFGNodesMap.get(label + ":" + lineNum);
            Set<DFVarNode> currentDefVarMap = new HashSet<>();
            NodeList<Parameter> parameters = methodDeclaration.getParameters();
            for (Parameter parameter : parameters) {
                currentDefVarMap.add(new DFVarNode(parameter.getNameAsString(), methodNode));
            }

            Optional<BlockStmt> body = methodDeclaration.getBody();
            if (body.isPresent()) {
                NodeList<Statement> statements = body.get().getStatements();
                for (Statement statement : statements) {
                    currentDefVarMap = buildDFG(statement, currentDefVarMap);
                }
            }
        }
    }


    /**
     * @param node            当前statement
     * @param parentDefVarMap 当前statement前一个statement的变量集
     * @return
     */
    private Set<DFVarNode> buildDFG(Node node, Set<DFVarNode> parentDefVarMap) {

        if (node instanceof ExpressionStmt) {
            ExpressionStmt expressionStmt = ((ExpressionStmt) node).asExpressionStmt();
            Expression expression = expressionStmt.getExpression();
            String label = expression.toString();

            if (LogUtil.isLogStatement(label, 1)) {
                return parentDefVarMap;
            }

            int lineNum = expression.getBegin().isPresent() ? expression.getBegin().get().line : -1;
            GraphNode cfgNode = allCFGNodesMap.get(label + ":" + lineNum);
            return dealSingleRoadStmtDFG(parentDefVarMap, expression, cfgNode);

        } else if (node instanceof ReturnStmt) {

            ReturnStmt returnStmt = ((ReturnStmt) node).asReturnStmt();
            String label = returnStmt.getExpression().isPresent() ? "return " + returnStmt.getExpression().get().toString() : "return";
            int lineNum = returnStmt.getBegin().isPresent() ? returnStmt.getBegin().get().line : -1;
            if (returnStmt.getExpression().isPresent()) {
                GraphNode cfgNode = allCFGNodesMap.get(label + ":" + lineNum);
                parentDefVarMap = dealSingleRoadStmtDFG(parentDefVarMap, returnStmt.getExpression().get(), cfgNode);
            }
            return parentDefVarMap;
        } else if (node instanceof AssertStmt) {
            AssertStmt assertStmt = ((AssertStmt) node).asAssertStmt();
            String label = assertStmt.getMessage().isPresent() ? "assert" + assertStmt.getCheck().toString() + ";" + assertStmt.getMessage().get().toString() : "assert" + assertStmt.getCheck().toString();
            int lineNum = assertStmt.getBegin().isPresent() ? assertStmt.getBegin().get().line : -1;
            GraphNode cfgNode = allCFGNodesMap.get(label + ":" + lineNum);
            return dealSingleRoadStmtDFG(parentDefVarMap, assertStmt.getCheck(), cfgNode);

        } else if (node instanceof ThrowStmt) {
            ThrowStmt throwStmt = ((ThrowStmt) node).asThrowStmt();
            String label = "throw " + throwStmt.getExpression();
            int lineNum = throwStmt.getBegin().isPresent() ? throwStmt.getBegin().get().line : -1;
            GraphNode cfgNode = allCFGNodesMap.get(label + ":" + lineNum);
            return dealSingleRoadStmtDFG(parentDefVarMap, throwStmt.getExpression(), cfgNode);
        } else if (node instanceof IfStmt) {

            IfStmt tempIfStmt = ((IfStmt) node).asIfStmt();
            Set<DFVarNode> copy = this.copy(parentDefVarMap);
            while (tempIfStmt != null) {
                String ifLabel = "if (" + tempIfStmt.getCondition().toString() + ")";
                int ifLineNum = tempIfStmt.getBegin().isPresent() ? tempIfStmt.getBegin().get().line : -1;
                GraphNode ifCfgNode = allCFGNodesMap.get(ifLabel + ":" + ifLineNum);
                Set<DFVarNode> sonBlockDefVarSet = this.copy(copy);
                sonBlockDefVarSet = dealSingleRoadStmtDFG(sonBlockDefVarSet, tempIfStmt.getCondition(), ifCfgNode);
                if (!tempIfStmt.getThenStmt().isBlockStmt()) {
                    sonBlockDefVarSet = buildDFG(tempIfStmt.getThenStmt(), sonBlockDefVarSet);
                } else {
                    BlockStmt thenBlockStmt = tempIfStmt.getThenStmt().asBlockStmt();
                    NodeList<Statement> statements = thenBlockStmt.getStatements();
                    for (Statement statement : statements) {
                        sonBlockDefVarSet = buildDFG(statement, sonBlockDefVarSet);
                    }
                }

                parentDefVarMap = this.merge(parentDefVarMap, sonBlockDefVarSet);
                if (tempIfStmt.getElseStmt().isPresent()) {
                    if (tempIfStmt.getElseStmt().get().isIfStmt()) {
                        tempIfStmt = tempIfStmt.getElseStmt().get().asIfStmt();
                    } else {
                        Set<DFVarNode> sonBlockDefVarSet2 = this.copy(copy);
                        if (!tempIfStmt.getElseStmt().get().isBlockStmt()) {
                            sonBlockDefVarSet2 = buildDFG(tempIfStmt.getElseStmt().get(), sonBlockDefVarSet2);
                        } else {
                            BlockStmt elseBlockStmt = tempIfStmt.getElseStmt().get().asBlockStmt();
                            NodeList<Statement> statements1 = elseBlockStmt.getStatements();
                            for (Statement statement : statements1) {
                                sonBlockDefVarSet2 = buildDFG(statement, sonBlockDefVarSet2);
                            }
                        }
                        parentDefVarMap = this.merge(parentDefVarMap, sonBlockDefVarSet2);

                        tempIfStmt = null;
                    }
                } else {
                    tempIfStmt = null;
                }
            }
            return parentDefVarMap;
        } else if (node instanceof WhileStmt) {
            WhileStmt whileStmt = ((WhileStmt) node).asWhileStmt();
            String label = "while (" + whileStmt.getCondition().toString() + ")";
            int lineNum = whileStmt.getBegin().isPresent() ? whileStmt.getBegin().get().line : -1;
            GraphNode cfgNode = allCFGNodesMap.get(label + ":" + lineNum);
            parentDefVarMap = dealSingleRoadStmtDFG(parentDefVarMap, whileStmt.getCondition(), cfgNode);
            Set<DFVarNode> sonBlockDefVarSet = this.copy(parentDefVarMap);
            if (!whileStmt.getBody().isBlockStmt()) {
                sonBlockDefVarSet = buildDFG(whileStmt.getBody(), sonBlockDefVarSet);
            } else {
                NodeList<Statement> statements = whileStmt.getBody().asBlockStmt().getStatements();
                if (statements.size() == 0) {
                    return parentDefVarMap;
                }
                for (Statement statement : statements) {
                    sonBlockDefVarSet = buildDFG(statement, sonBlockDefVarSet);
                }
            }
            parentDefVarMap = this.merge(parentDefVarMap, sonBlockDefVarSet);
            return parentDefVarMap;
        } else if (node instanceof ForStmt) {
            List<String> forValues = new ArrayList<>();
            Set<DFVarNode> sonBlockDefVarSet = this.copy(parentDefVarMap);
            ForStmt forStmt = ((ForStmt) node).asForStmt();
            forValues.add(StringUtils.join(forStmt.getInitialization(), ","));
            if (forStmt.getCompare().isPresent()) {
                forValues.add(forStmt.getCompare().get().toString());
            }
            forValues.add(StringUtils.join(forStmt.getUpdate(), ","));
            String label = "for(" + StringUtils.join(forValues, ';') + ")";
            int lineNum = forStmt.getBegin().isPresent() ? forStmt.getBegin().get().line : -1;
            GraphNode cfgNode = allCFGNodesMap.get(label + ":" + lineNum);
            NodeList<Expression> initialization = forStmt.getInitialization();
            for (Expression e : initialization) {
                sonBlockDefVarSet = dealSingleRoadStmtDFG(sonBlockDefVarSet, e, cfgNode);
            }
            if (!forStmt.getBody().isBlockStmt()) {
                sonBlockDefVarSet = buildDFG(forStmt.getBody(), sonBlockDefVarSet);
            } else {
                NodeList<Statement> statements = forStmt.getBody().asBlockStmt().getStatements();
                if (statements.size() == 0) {
                    return parentDefVarMap;
                }
                for (Statement statement : statements) {
                    sonBlockDefVarSet = buildDFG(statement, sonBlockDefVarSet);
                }
            }
            parentDefVarMap = this.merge(parentDefVarMap, sonBlockDefVarSet);
            return parentDefVarMap;
        } else if (node instanceof ForeachStmt) {
            ForeachStmt foreachStmt = ((ForeachStmt) node).asForeachStmt();
            Set<DFVarNode> sonBlockDefVarSet = this.copy(parentDefVarMap);
            String label = "for(" + foreachStmt.getVariable() + ":" + foreachStmt.getIterable() + ")";
            int lineNum = foreachStmt.getBegin().isPresent() ? foreachStmt.getBegin().get().line : -1;
            GraphNode cfgNode = allCFGNodesMap.get(label + ":" + lineNum);
            sonBlockDefVarSet = dealSingleRoadStmtDFG(sonBlockDefVarSet, foreachStmt.getVariable(), cfgNode);
            sonBlockDefVarSet = dealSingleRoadStmtDFG(sonBlockDefVarSet, foreachStmt.getIterable(), cfgNode);
            if (!foreachStmt.getBody().isBlockStmt()) {
                sonBlockDefVarSet = buildDFG(foreachStmt.getBody(), sonBlockDefVarSet);
            } else {
                NodeList<Statement> statements = foreachStmt.getBody().asBlockStmt().getStatements();
                if (statements.size() == 0) {
                    return parentDefVarMap;
                }
                for (Statement statement : statements) {
                    sonBlockDefVarSet = buildDFG(statement, sonBlockDefVarSet);
                }
            }
            parentDefVarMap = this.merge(parentDefVarMap, sonBlockDefVarSet);
            return parentDefVarMap;
        } else if (node instanceof SwitchStmt) {
            SwitchStmt switchStmt = ((SwitchStmt) node).asSwitchStmt();
            String label = "switch(" + switchStmt.getSelector().toString() + ")";
            int lineNum = switchStmt.getBegin().isPresent() ? switchStmt.getBegin().get().line : -1;
            GraphNode cfgNode = allCFGNodesMap.get(label + ":" + lineNum);
            parentDefVarMap = dealSingleRoadStmtDFG(parentDefVarMap, switchStmt.getSelector(), cfgNode);
            NodeList<SwitchEntryStmt> caseEntries = switchStmt.getEntries(); //case 入口
            if (caseEntries.size() == 0) {
                return parentDefVarMap;
            }
            Set<DFVarNode> copy = this.copy(parentDefVarMap);
            for (int i = 0; i < caseEntries.size(); i++) {
                Set<DFVarNode> sonBlockDefVarSet = this.copy(copy);
                NodeList<Statement> statements = caseEntries.get(i).getStatements();
                for (Statement statement : statements) {
                    sonBlockDefVarSet = buildDFG(statement, sonBlockDefVarSet);
                }
                parentDefVarMap = this.merge(parentDefVarMap, sonBlockDefVarSet);
            }
            return parentDefVarMap;
        } else if (node instanceof DoStmt) {
            DoStmt doStmt = ((DoStmt) node).asDoStmt();
            Set<DFVarNode> sonBlockDefVarSet = this.copy(parentDefVarMap);

            if (!doStmt.getBody().isBlockStmt()) {
                sonBlockDefVarSet = buildDFG(doStmt.getBody(), sonBlockDefVarSet);
            } else {
                NodeList<Statement> statements = doStmt.getBody().asBlockStmt().getStatements();
                for (Statement statement : statements) {
                    sonBlockDefVarSet = buildDFG(statement, sonBlockDefVarSet);
                }
            }
            String label = "while (" + doStmt.getCondition().toString() + ")";
            int lineNum = doStmt.getCondition().getBegin().isPresent() ? doStmt.getCondition().getBegin().get().line : -1;
            GraphNode cfgNode = allCFGNodesMap.get(label + ":" + lineNum);
            sonBlockDefVarSet = dealSingleRoadStmtDFG(sonBlockDefVarSet, doStmt.getCondition(), cfgNode);
            parentDefVarMap = this.merge(parentDefVarMap, sonBlockDefVarSet);
            return parentDefVarMap;
        } else if (node instanceof LabeledStmt) {
            LabeledStmt labeledStmt = ((LabeledStmt) node).asLabeledStmt();
            buildDFG(labeledStmt.getStatement(), parentDefVarMap);
        } else if (node instanceof SynchronizedStmt) {
            SynchronizedStmt synchronizedStmt = ((SynchronizedStmt) node).asSynchronizedStmt();
            String label = "synchronized (" + synchronizedStmt.getExpression() + ")";
            int lineNum = synchronizedStmt.getBegin().isPresent() ? synchronizedStmt.getBegin().get().line : -1;
            GraphNode cfgNode = allCFGNodesMap.get(label + ":" + lineNum);
            parentDefVarMap = dealSingleRoadStmtDFG(parentDefVarMap, synchronizedStmt.getExpression(), cfgNode);
            BlockStmt body = synchronizedStmt.getBody();
            NodeList<Statement> statements = body.getStatements();
            for (Statement statement : statements) {
                parentDefVarMap = buildDFG(statement, parentDefVarMap);
            }
            return parentDefVarMap;
        } else if (node instanceof BlockStmt) {
            BlockStmt blockStmt = ((BlockStmt) node).asBlockStmt();
            NodeList<Statement> statements = blockStmt.getStatements();
            for (Statement statement : statements) {
                parentDefVarMap = buildDFG(statement, parentDefVarMap);
            }
            return parentDefVarMap;
        } else if (node instanceof TryStmt) {
            TryStmt tryStmt = ((TryStmt) node).asTryStmt();
            String label = tryStmt.getResources().size() == 0 ? "try" : "try(" + StringUtils.join(tryStmt.getResources(), ";") + ")";
            int lineNum = tryStmt.getBegin().isPresent() ? tryStmt.getBegin().get().line : -1;
            GraphNode cfgNode = allCFGNodesMap.get(label + ":" + lineNum);
            NodeList<Expression> resources = tryStmt.getResources();
            for (Expression e : resources) {
                parentDefVarMap = dealSingleRoadStmtDFG(parentDefVarMap, e, cfgNode);
            }

            BlockStmt tryBlock = tryStmt.getTryBlock();
            NodeList<Statement> statements = tryBlock.getStatements();
            for (Statement statement : statements) {
                parentDefVarMap = buildDFG(statement, parentDefVarMap);
            }
            Optional<BlockStmt> finallyBlock = tryStmt.getFinallyBlock();
            if (finallyBlock.isPresent()) {
                NodeList<Statement> finaBodyStas = finallyBlock.get().getStatements();
                for (Statement statement : finaBodyStas) {
                    parentDefVarMap = buildDFG(statement, parentDefVarMap);
                }
            }
            return parentDefVarMap;
        } else if (node instanceof LocalClassDeclarationStmt) {
            LocalClassDeclarationStmt localClassDeclarationStmt = ((LocalClassDeclarationStmt) node).asLocalClassDeclarationStmt();
            ClassOrInterfaceDeclaration classOrInterfaceDeclaration = localClassDeclarationStmt.getClassDeclaration();
            String label = classOrInterfaceDeclaration.getNameAsString();
            int lineNum = classOrInterfaceDeclaration.getBegin().isPresent() ? classOrInterfaceDeclaration.getBegin().get().line : -1;
            GraphNode cfgNode = allCFGNodesMap.get(label + ":" + lineNum);
        }
        return parentDefVarMap;
    }

    /**
     * 解析表达式得到这个表达式所使用的变量，定义的变量直接通过创建来的 List<HashMap<String,GraphNode>> parentDefVarMap
     * 存储起来，所以只需要返回这个expression 所用过的变量的信息
     *
     * @return list[0] 存储使用的变量 ，list[1] 存储定义的变量  因为更新定义的变量必须在使用的变量边创建之后才行
     */
    private List<Set<String>> analysisExprForVar(Expression expression) {
        if (expression instanceof ArrayAccessExpr) {
            /*
             ArrayAccessExpr 就是获取数组值表达式 比如datas[a]
             */
            ArrayAccessExpr arrayAccessExpr = expression.asArrayAccessExpr();
            Expression name = arrayAccessExpr.getName();
            Expression index = arrayAccessExpr.getIndex();
            List<Set<String>> sets = analysisExprForVar(name);
            List<Set<String>> sets1 = analysisExprForVar(index);
            // 将使用过的变量和定义的变量都存储起来
            sets.get(0).addAll(sets1.get(0));
            sets.get(1).addAll(sets1.get(1));
            return sets;
        } else if (expression instanceof ClassExpr) {
            /*
            ClassExpr Object.class 一个类获取class对象
             */
            ClassExpr classExpr = expression.asClassExpr();
        } else if (expression instanceof ArrayCreationExpr) {
            /*
            ArrayCreationExpr new int[5] 5 可能变成其他变量 所以可能改变数据流
             */
            ArrayCreationExpr arrayCreationExpr = expression.asArrayCreationExpr();
            NodeList<ArrayCreationLevel> levels = arrayCreationExpr.getLevels();
            List<Set<String>> result = new ArrayList<>();
            Set<String> s0 = new HashSet<>();
            Set<String> s1 = new HashSet<>();
            result.add(s0);
            result.add(s1);
            for (ArrayCreationLevel a : levels) {
                //把数组创建中维度是变量的节点信息记住！
                if (a.getDimension().isPresent()) {
                    List<Set<String>> sets = analysisExprForVar(a.getDimension().get());
                    result.get(0).addAll(sets.get(0));
                    result.get(1).addAll(sets.get(1));
                }
            }
            if (arrayCreationExpr.getInitializer().isPresent()) {
                ArrayInitializerExpr arrayInitializerExpr = arrayCreationExpr.getInitializer().get();
                NodeList<Expression> values = arrayInitializerExpr.getValues();
                for (Expression expression1 : values) {
                    List<Set<String>> sets = analysisExprForVar(expression1);
                    result.get(0).addAll(sets.get(0));
                    result.get(1).addAll(sets.get(1));
                }
            }
            return result;

        } else if (expression instanceof LambdaExpr) {
            /*
            lambda (a, b) -> a+b 这是定义函数的方式 所以对于数据流没有任何帮助
             */
            LambdaExpr lambdaExpr = expression.asLambdaExpr();
        } else if (expression instanceof ConditionalExpr) {
            /*
             条件表达式 比如 if(a) 也就是里面有用 ifstmt中的if 包含在这个expr里面
             */
            ConditionalExpr conditionalExpr = expression.asConditionalExpr();
            List<Set<String>> sets = analysisExprForVar(conditionalExpr.getCondition());
            List<Set<String>> sets1 = analysisExprForVar(conditionalExpr.getThenExpr());
            List<Set<String>> sets2 = analysisExprForVar(conditionalExpr.getElseExpr());
            sets.get(0).addAll(sets1.get(0));
            sets.get(1).addAll(sets1.get(1));
            sets.get(0).addAll(sets2.get(0));
            sets.get(1).addAll(sets2.get(1));
            return sets;
        } else if (expression instanceof MethodCallExpr) {
            /*
            MethodCallExpr System.out.println("true");
             */
            MethodCallExpr methodCallExpr = expression.asMethodCallExpr();
            List<Set<String>> result = new ArrayList<>();
            Set<String> s0 = new HashSet<>();
            Set<String> s1 = new HashSet<>();
            result.add(s0);
            result.add(s1);
            //这个是得到方法的调用变量
            if (methodCallExpr.getScope().isPresent()) {
                List<Set<String>> sets = analysisExprForVar(methodCallExpr.getScope().get());
                result.get(0).addAll(sets.get(0));
                result.get(1).addAll(sets.get(1));
            }
            //继续拿到方法的参数变量名字
            NodeList<Expression> arguments = methodCallExpr.getArguments();
            for (Expression expression1 : arguments) {
                List<Set<String>> sets = analysisExprForVar(expression1);
                result.get(0).addAll(sets.get(0));
                result.get(1).addAll(sets.get(1));
            }
            return result;
        } else if (expression instanceof AnnotationExpr) {
            /*
            对数据流没有任何影响 这是方法的注解
             */
            AnnotationExpr annotationExpr = expression.asAnnotationExpr();

        } else if (expression instanceof AssignExpr) {
            /*
            赋值表达式   datas[0] = 1;
             */
            AssignExpr assignExpr = expression.asAssignExpr();
            List<Set<String>> sets = analysisExprForVar(assignExpr.getTarget());
            sets.get(1).addAll(sets.get(0));
            //注意上面是赋值操作的左边 所以这些变量 1.是赋值变量 2.是使用变量 依赖于operator
//            if(!assignExpr.getOperator().equals(AssignExpr.Operator.ASSIGN)){
//                //左边的变量是使用变量的同时也是定义变量 默认的name 里面都是使用变量
//                sets.get(1).addAll(sets.get(0));
//            }else{
//                //assign 就是定义变量
//                for(String s:sets.get(0)){
//                    sets.get(1).add(s);
//                }
//                //清空定义变量
//                sets.get(0).clear();
//            }

            // 赋值语句右边就是使用变量
            List<Set<String>> sets1 = analysisExprForVar(assignExpr.getValue());

            sets.get(0).addAll(sets1.get(0));
            sets.get(1).addAll(sets1.get(1));

            return sets;

        } else if (expression instanceof InstanceOfExpr) {
            /*
            instance of 对于数据流 有影响就是用数据流
             */
            InstanceOfExpr instanceOfExpr = expression.asInstanceOfExpr();
            return analysisExprForVar(instanceOfExpr.getExpression());
        } else if (expression instanceof CastExpr) {
            /*
            caseExpr  对于数据流没有任何影响 (long)15 long数字
             */
            CastExpr castExpr = expression.asCastExpr();
            return analysisExprForVar(castExpr.getExpression());
        } else if (expression instanceof NameExpr) {
            /*
            变量的名字  switch(a) 里面的a
             */
            NameExpr nameExpr = expression.asNameExpr();
            List<Set<String>> result = new ArrayList<>();
            Set<String> s0 = new HashSet<>();
            Set<String> s1 = new HashSet<>();
            //默认解析到NameExpr的时候 都是来源于使用的，对于Assign单独处理
            s0.add(nameExpr.getName().getIdentifier());
            result.add(s0);
            result.add(s1);
            return result;

        } else if (expression instanceof ThisExpr) {
            /*
            this 字符 对于数据流没有任何影响
             */
            ThisExpr thisExpr = expression.asThisExpr();
            if (thisExpr.getClassExpr().isPresent()) {
                return analysisExprForVar(thisExpr.getClassExpr().get());
            }
        } else if (expression instanceof EnclosedExpr) {
            /*
              括号内的表达式 (1+1)
             */
            EnclosedExpr enclosedExpr = expression.asEnclosedExpr();
            List<Set<String>> sets = analysisExprForVar(enclosedExpr.getInner());
            return sets;

        } else if (expression instanceof MethodReferenceExpr) {
            /*
             方法引用 左边是对象 System.out::println 的println
             */
            MethodReferenceExpr methodReferenceExpr = expression.asMethodReferenceExpr();
            return analysisExprForVar(methodReferenceExpr.getScope());

        } else if (expression instanceof VariableDeclarationExpr) {
            /*
            VariableDeclarator 是 VariableDeclarationExpr 更细的粒度
            int[] datas = { 1, 2, 3, 4 } int a = 10
             */
            //只有这个节点才是变量定义节点 所以需要记录当前节点的变量定义信息！
            VariableDeclarationExpr variableDeclarationExpr = expression.asVariableDeclarationExpr();
            NodeList<VariableDeclarator> variables = variableDeclarationExpr.getVariables();
            List<Set<String>> result = new ArrayList<>();

            // s1里存储用到的变量
            // s2里存储定义的变量
            Set<String> s0 = new HashSet<>();
            Set<String> s1 = new HashSet<>();
            result.add(s0);
            result.add(s1);

            for (VariableDeclarator var : variables) {
                s1.add(var.getNameAsString());
                if (var.getInitializer().isPresent()) {
                    // var就是定义的节点
                    // var.getInitializer()是var的表达式
                    List<Set<String>> sets = analysisExprForVar(var.getInitializer().get());
                    result.get(0).addAll(sets.get(0));
                    result.get(1).addAll(sets.get(1));
                }
            }
            return result;
        } else if (expression instanceof LiteralExpr) {
            /*
            文字表达式 也就是null true 等数值 对于数据流毫无影响
             */
            LiteralExpr literalExpr = expression.asLiteralExpr();

        } else if (expression instanceof ObjectCreationExpr) {
            /*
            new Exception("yichang") 声明变量的后一半
             */
            ObjectCreationExpr objectCreationExpr = expression.asObjectCreationExpr();

            List<Set<String>> result = new ArrayList<>();
            Set<String> s0 = new HashSet<>();
            Set<String> s1 = new HashSet<>();
            result.add(s0);
            result.add(s1);
            NodeList<Expression> arguments = objectCreationExpr.getArguments();
            for (Expression expression1 : arguments) {
                List<Set<String>> sets = analysisExprForVar(expression1);
                result.get(0).addAll(sets.get(0));
                result.get(1).addAll(sets.get(1));
            }
            return result;
        } else if (expression instanceof UnaryExpr) {
            /*
            一元运算符 i++
             */
            UnaryExpr unaryExpr = expression.asUnaryExpr();
            List<Set<String>> sets = analysisExprForVar(unaryExpr.getExpression());
            //assign 就是定义变量
            UnaryExpr.Operator operator = unaryExpr.getOperator();
            if (operator.equals(UnaryExpr.Operator.PREFIX_DECREMENT) || operator.equals(UnaryExpr.Operator.PREFIX_INCREMENT) || operator.equals(UnaryExpr.Operator.POSTFIX_DECREMENT) || operator.equals(UnaryExpr.Operator.POSTFIX_INCREMENT)) {
                //这四种符号的变量 既充当使用的变量 也充当赋值的变量
                sets.get(1).addAll(sets.get(0));
            }
            return sets;
        } else if (expression instanceof SuperExpr) {
            /*
               SuperExpr  super 这个字符 对于数据流影响
             */
            SuperExpr superExpr = expression.asSuperExpr();
            if (superExpr.getClassExpr().isPresent()) {
                return analysisExprForVar(superExpr.getClassExpr().get());
            }
        } else if (expression instanceof BinaryExpr) {
            /*
            二元操作符表达式 比如if 条件中 a==10
             */
            BinaryExpr binaryExpr = expression.asBinaryExpr();
            List<Set<String>> sets = analysisExprForVar(binaryExpr.getLeft());
            List<Set<String>> sets1 = analysisExprForVar(binaryExpr.getRight());
            sets.get(0).addAll(sets1.get(0));
            sets.get(1).addAll(sets1.get(1));
            return sets;
        } else if (expression instanceof TypeExpr) {
            /*
            方法引用 World::greet 的world 就是类型 类名字
             */
            TypeExpr typeExpr = expression.asTypeExpr();

        } else if (expression instanceof ArrayInitializerExpr) {
            /*
            new int[][] {{1, 1}, {2, 2}}
             */
            ArrayInitializerExpr arrayInitializerExpr = expression.asArrayInitializerExpr();
            List<Set<String>> result = new ArrayList<>();
            Set<String> s0 = new HashSet<>();
            Set<String> s1 = new HashSet<>();
            result.add(s0);
            result.add(s1);
            NodeList<Expression> values = arrayInitializerExpr.getValues();
            for (Expression expression1 : values) {
                List<Set<String>> sets = analysisExprForVar(expression1);
                result.get(0).addAll(sets.get(0));
                result.get(1).addAll(sets.get(1));
            }
            return result;
        } else if (expression instanceof FieldAccessExpr) {
            /*
            对象获取属性 FieldAccessExpr person.name
             */
            FieldAccessExpr fieldAccessExpr = expression.asFieldAccessExpr();
            List<Set<String>> sets = analysisExprForVar(fieldAccessExpr.getScope());
            return sets;
        }
        Set<String> s1 = new HashSet<>();
        Set<String> s2 = new HashSet<>();
        List<Set<String>> result = new ArrayList<>();
        result.add(s1);
        result.add(s2);
        return result;
    }

    /**
     * 代码中很多单线路的stmt 这些stmt的更新方式就先先把expression的引用的变量连上边，然后就是更新定义变量的信息
     *
     * @param expression 需要分析的源码表达式
     * @param node       被分析源码所在的node的信息
     * @return 更新之后的定义的变量信息
     */
    private Set<DFVarNode> dealSingleRoadStmtDFG(Set<DFVarNode> parentDefVarMap, Expression expression, GraphNode node) {

        List<Set<String>> varsUsed = analysisExprForVar(expression);
        // s0 用到的变量
        // s1 被赋值的变量 可能是第一次定义(int a = 0) 也可能是定义过但被重新赋值了(a = 0)
        Set<String> s0 = varsUsed.get(0);
        Set<String> s1 = varsUsed.get(1);

        // 当前节点连接到依赖的变量
        for (String varName : s0) {
            List<DFVarNode> tempNodes = getDefination(parentDefVarMap, varName);
            for (DFVarNode tempNode : tempNodes) {
                connectNodes(node, tempNode.getNode());
            }
        }

        // 更新定义
        for (String varName : s1) {
            intersection(parentDefVarMap, varName, node);
        }
        return parentDefVarMap;
    }


    /**
     * 单线路的节点 数据流就是取交集 保留最新节点信息
     *
     * @param in      表示到达节点之前的变量定义的信息
     * @param varName 变量的名字信息
     * @return 返回去除了除了当前这个变量之前的所有该变量，因为是单线路
     */
    private Set<DFVarNode> intersection(Set<DFVarNode> in, String varName, GraphNode node) {
        List<DFVarNode> tempNodes = getDefination(in, varName);
        if (!tempNodes.isEmpty()) {
            // 已经被定义过了 更新即可
            for (DFVarNode tempNode : tempNodes) {
                tempNode.setNode(node);
            }
        } else {
            DFVarNode dfVarNode = new DFVarNode(varName, node);
            in.add(dfVarNode);
        }
        return in;
    }

    /**
     * 多线路的节点 数据流就是取并集 保留多条路径信息
     *
     * @param in1 第一个分支出来的变量定义的信息
     * @param in2 第二个分支出来的变量定义的信息
     * @return 返回所有分支并起来的数据信息
     */
    private Set<DFVarNode> merge(Set<DFVarNode> in1, Set<DFVarNode> in2) {
        Set<DFVarNode> result = new HashSet<>();
        result.addAll(in1);
        result.addAll(in2);
        return result;
    }

    /**
     * 每一个局部作用域 都有自己的作用范围 超过这个范围，在该范围内定义的变量都会被消除
     *
     * @param in 父范围定义的变量信息
     * @return 返回只是copy这个父范围的变量信息的一样的数据结构对象
     */
    private Set<DFVarNode> copy(Set<DFVarNode> in) {
        Set<DFVarNode> result = new HashSet<>();
        // 浅拷贝每个节点
        for (DFVarNode node : in) {
            DFVarNode clone = (DFVarNode) node.clone();
            result.add(clone);
        }
        return result;
    }

    public Set<GraphEdge> getAllDFGEdgesList() {
        return allDFGEdgesList;
    }

    public void setAllDFGEdgesList(Set<GraphEdge> allDFGEdgesList) {
        this.allDFGEdgesList = allDFGEdgesList;
    }


    /**
     * 判断变量在DefVarMap中是否存在
     *
     * @param DefVarMap
     * @param varName
     * @return
     */
    private List<DFVarNode> getDefination(Set<DFVarNode> DefVarMap, String varName) {
        List<DFVarNode> result = new ArrayList<>();
        for (DFVarNode node : DefVarMap) {
            if (node.getVarName().equals(varName))
                result.add(node);
        }
        return result;
    }

    /**
     * 使用DFG边连接两个节点，前面一个指向后面一个
     *
     * @param originalNode
     * @param aimNode
     */
    private void connectNodes(GraphNode originalNode, GraphNode aimNode) {
        GraphEdge DFGEdge = new GraphEdge(EdgeTypes.DFG, originalNode, aimNode);
        allDFGEdgesList.add(DFGEdge);
    }

}
