package model;

import com.github.javaparser.ast.Node;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;


public class AstNode implements Serializable {
    private static final long serialVersionUID = 1L;
    private String name;
    private String typeName;
    private int lineBegin;
    private Node rootPrimary;


    private List<String> attributes;


    private List<AstNode> subNodes;


    private List<Node> subNodesPrimary;

    private List<String> subLists;

    private List<String> subLists_name;

    private List<List<AstNode>> subListNodes;

    private List<List<Node>> subListNodesPrimary;

    public AstNode() {
        this.attributes = new ArrayList<String>();
        this.subNodes = new ArrayList<AstNode>();
        this.subLists = new ArrayList<String>();
        this.subListNodes = new ArrayList<List<AstNode>>();
        this.subLists_name = new ArrayList<String>();
        this.subListNodesPrimary = new ArrayList<List<Node>>();
        this.subNodesPrimary = new ArrayList<Node>();
    }

    public String getTypeName() {
        return typeName;
    }

    public List<String> getAttributes() {
        return attributes;
    }

    public List<AstNode> getSubNodes() {
        return subNodes;
    }

    public void setTypeName(String typeName) {
        this.typeName = typeName;
    }

    public void setSubNodes(List<AstNode> subNodes) {
        this.subNodes = subNodes;
    }

    public List<String> getSubLists() {
        return subLists;
    }

    public List<List<AstNode>> getSubListNodes() {
        return subListNodes;
    }

    public void setSubLists(List<String> subLists) {
        this.subLists = subLists;
    }

    public void setSubListNodes(List<List<AstNode>> subListNodes) {
        this.subListNodes = subListNodes;
    }

    public String getName() {
        return name;
    }

    public List<String> getSubLists_name() {
        return subLists_name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public void setSubLists_name(List<String> subLists_name) {
        this.subLists_name = subLists_name;
    }

    public int getLineBegin() {
        return lineBegin;
    }

    public void setLineBegin(int lineBegin) {
        this.lineBegin = lineBegin;
    }

    public List<Node> getSubNodesPrimary() {
        return subNodesPrimary;
    }

    public List<List<Node>> getSubListNodesPrimary() {
        return subListNodesPrimary;
    }

    public void setAttributes(List<String> attributes) {
        this.attributes = attributes;
    }

    public void setSubNodesPrimary(List<Node> subNodesPrimary) {
        this.subNodesPrimary = subNodesPrimary;
    }

    public void setSubListNodesPrimary(List<List<Node>> subListNodesPrimary) {
        this.subListNodesPrimary = subListNodesPrimary;
    }

    public Node getRootPrimary() {
        return rootPrimary;
    }

    public void setRootPrimary(Node rootPrimary) {
        this.rootPrimary = rootPrimary;
    }

    @Override
    public String toString() {
        return "AstNode{" +
                "name='" + name + '\'' +
                ", typeName='" + typeName + '\'' +
                ", lineBegin=" + lineBegin +
                ", rootPrimary=" + rootPrimary +
                ", attributes=" + attributes +
                ", subNodes=" + subNodes +
                ", subNodesPrimary=" + subNodesPrimary +
                ", subLists=" + subLists +
                ", subLists_name=" + subLists_name +
                ", subListNodes=" + subListNodes +
                ", subListNodesPrimary=" + subListNodesPrimary +
                '}';
    }
}
