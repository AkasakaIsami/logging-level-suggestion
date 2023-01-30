package model;

import java.util.Objects;


public class DFVarNode implements Cloneable {
    private String varName;
    private GraphNode node;
    private String varType;

    public DFVarNode() {

    }

    public DFVarNode(String varName, GraphNode node) {
        this.varName = varName;
        this.node = node;
    }

    public DFVarNode(String varName, GraphNode node, String varType) {
        this.varName = varName;
        this.node = node;
        this.varType = varType;
    }

    public DFVarNode(String varName, String varType) {
        this.varName = varName;
        this.varType = varType;
    }

    public String getVarName() {
        return varName;
    }

    public void setVarName(String varName) {
        this.varName = varName;
    }

    public GraphNode getNode() {
        return node;
    }

    public void setNode(GraphNode node) {
        this.node = node;
    }

    public String getVarType() {
        return varType;
    }

    public void setVarType(String varType) {
        this.varType = varType;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        DFVarNode dfVarNode = (DFVarNode) o;
        return Objects.equals(varName, dfVarNode.varName) &&
                Objects.equals(node, dfVarNode.node);
    }

    @Override
    public int hashCode() {
        return Objects.hash(varName, node);
    }

    @Override
    public String toString() {
        return "DFVarNode{" +
                "varName='" + varName + '\'' +
                ", node=" + node +
                ", varType='" + varType + '\'' +
                '}';
    }

    public Object clone() {
        Object obj = null;
        try {
            obj = super.clone();
        } catch (CloneNotSupportedException e) {
            e.printStackTrace();
        }
        return obj;
    }

}
