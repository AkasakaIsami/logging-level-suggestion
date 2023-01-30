package service;

import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.NodeList;
import com.github.javaparser.ast.comments.Comment;
import com.github.javaparser.ast.stmt.BlockStmt;
import com.github.javaparser.ast.stmt.CatchClause;
import com.github.javaparser.ast.stmt.IfStmt;
import com.github.javaparser.metamodel.NodeMetaModel;
import com.github.javaparser.metamodel.PropertyMetaModel;
import model.AstNode;

import java.util.ArrayList;
import java.util.List;

import static com.github.javaparser.utils.Utils.assertNotNull;
import static java.util.stream.Collectors.toList;


public class AstNodeInit {

    private final boolean outputNodeType;
    private AstNode astNode;

    public AstNodeInit(boolean outputNodeType, AstNode astNode) {
        this.outputNodeType = outputNodeType;
        this.astNode = astNode;
    }

    public void Init(Node node) {
        astNode.setRootPrimary(node);
        output(node, "root", astNode);
    }



    private void output(Node node, String name, AstNode astNode) {
        assertNotNull(node);
        NodeMetaModel metaModel = node.getMetaModel();
        List<PropertyMetaModel> allPropertyMetaModels = metaModel.getAllPropertyMetaModels();

        // 所有是属性的属性
        List<PropertyMetaModel> attributes = allPropertyMetaModels.stream().filter(PropertyMetaModel::isAttribute).filter(PropertyMetaModel::isSingular).collect(toList());

        // 所有是单个node的属性
        List<PropertyMetaModel> subNodes = allPropertyMetaModels.stream().filter(PropertyMetaModel::isNode).filter(PropertyMetaModel::isSingular).collect(toList());

        // 所有是nodelist的属性
        List<PropertyMetaModel> subLists = allPropertyMetaModels.stream().filter(PropertyMetaModel::isNodeList).collect(toList());

        if (outputNodeType) {
            astNode.setTypeName(name + " (" + metaModel.getTypeName() + ")");
        } else {
            astNode.setTypeName(name);
        }


        for (PropertyMetaModel a : attributes) {
            astNode.getAttributes().add(a.getName() + "='" + a.getValue(node).toString() + "'");
        }

        for (PropertyMetaModel sn : subNodes) {
            Node nd = (Node) sn.getValue(node);

            if (nd != null && !nd.toString().equals("") && !(nd instanceof BlockStmt) && !(nd instanceof Comment) && !(nd instanceof IfStmt)) {
                AstNode subAstNode = new AstNode();
                astNode.getSubNodes().add(subAstNode);
                astNode.setName(sn.getName());
                astNode.getSubNodesPrimary().add(nd);
                output(nd, sn.getName(), subAstNode);
            }
        }

        for (PropertyMetaModel sl : subLists) {
            @SuppressWarnings("unchecked")
            NodeList<? extends Node> nl = (NodeList<? extends Node>) sl.getValue(node);
            if (nl != null && nl.isNonEmpty()) {
                astNode.getSubLists().add(sl.getName());
                String slName = sl.getName().substring(0, sl.getName().length() - 1);
                astNode.getSubLists_name().add(slName);
                List<AstNode> astNodes = new ArrayList<AstNode>();
                List<Node> primaryNodes = new ArrayList<Node>();
                astNode.getSubListNodes().add(astNodes);
                astNode.getSubListNodesPrimary().add(primaryNodes);
                for (Node nd : nl) {
                    if (!nd.toString().equals("") && !(nd instanceof BlockStmt) && !(nd instanceof CatchClause) && !(nd instanceof IfStmt)) {
                        primaryNodes.add(nd);
                        AstNode subAstNode = new AstNode();
                        astNodes.add(subAstNode);
                        output(nd, slName, subAstNode);
                    }
                }
            }
        }


    }


}
