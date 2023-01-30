package model;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;


public class GraphNode {


    private String originalCodeStr;
    private String opTypeStr;
    private String simplifyCodeStr;
    private String isExceptionLabel;
    private int codeLineNum;
    private String dotNum;




    private GraphNode parentNode;
    private List<GraphNode> adjacentPoints;
    private List<GraphEdge> edgs;
    private List<GraphNode> preAdjacentPoints;
    private AstNode astRootNode;

    {
        adjacentPoints = new ArrayList<>();
        edgs = new ArrayList<>();
        preAdjacentPoints = new ArrayList<>();
        isExceptionLabel = "false";
    }

    public GraphNode() {
    }

    public GraphNode(String originalCodeStr, String opTypeStr) {
        this.originalCodeStr = originalCodeStr;
        this.opTypeStr = opTypeStr;
    }

    public GraphNode(String originalCodeStr, String opTypeStr, String simplifyCodeStr, String isExceptionLabel) {
        this.originalCodeStr = originalCodeStr;
        this.opTypeStr = opTypeStr;
        this.simplifyCodeStr = simplifyCodeStr;
        this.isExceptionLabel = isExceptionLabel;
    }

    public String getOriginalCodeStr() {
        return originalCodeStr;
    }

    public void setOriginalCodeStr(String originalCodeStr) {
        this.originalCodeStr = originalCodeStr;
    }

    public String getOpTypeStr() {
        return opTypeStr;
    }

    public void setOpTypeStr(String opTypeStr) {
        this.opTypeStr = opTypeStr;
    }

    public String getSimplifyCodeStr() {
        return simplifyCodeStr;
    }

    public void setSimplifyCodeStr(String simplifyCodeStr) {
        this.simplifyCodeStr = simplifyCodeStr;
    }

    public String getIsExceptionLabel() {
        return isExceptionLabel;
    }

    public void setIsExceptionLabel(String isExceptionLabel) {
        this.isExceptionLabel = isExceptionLabel;
    }

    public List<GraphNode> getAdjacentPoints() {
        return adjacentPoints;
    }

    public void addAdjacentPoint(GraphNode adjacentPoint) {
        if(!this.adjacentPoints.contains(adjacentPoint)){
            this.adjacentPoints.add(adjacentPoint);
        }
    }


    public void removeAdjacentPoint(GraphNode adjacentPoint) {
        this.adjacentPoints.remove(adjacentPoint);
    }

    public List<GraphEdge> getEdgs() {
        return edgs;
    }

    public void addEdg(GraphEdge edge) {
        boolean insertFlug = true;
        for(GraphEdge e:this.edgs){
            if(e.getAimNode().equals(edge.getAimNode()) && e.getType().getColor().equals(edge.getType().getColor())){
                insertFlug = false;
                break;
            }
        }
        if(insertFlug){
            this.edgs.add(edge);
        }
    }

    public void removeEdges(GraphNode aimNode){
        Iterator<GraphEdge> iterator = this.edgs.iterator();
        while(iterator.hasNext()){
            GraphEdge next = iterator.next();
            if(next.getAimNode().equals(aimNode)){
                iterator.remove();
            }
        }
    }

    public int getCodeLineNum() {
        return codeLineNum;
    }

    public void setCodeLineNum(int codeLineNum) {
        this.codeLineNum = codeLineNum;
    }

    public GraphNode getParentNode() {
        return parentNode;
    }

    public void setParentNode(GraphNode parentNode) {
        this.parentNode = parentNode;
    }

    public AstNode getAstRootNode() {
        return astRootNode;
    }

    public void setAstRootNode(AstNode astRootNode) {
        this.astRootNode = astRootNode;
    }

    public String getDotNum() {
        return dotNum;
    }

    public void setDotNum(String dotNum) {
        this.dotNum = dotNum;
    }

    public List<GraphNode> getPreAdjacentPoints() {
        return preAdjacentPoints;
    }

    public void addPreAdjacentPoints(GraphNode preAdjacentPoint) {
        if(!this.preAdjacentPoints.contains(preAdjacentPoint)){
            this.preAdjacentPoints.add(preAdjacentPoint);
        }
    }

    public void removePreAdjacentPoints(GraphNode preAdjacentPoint) {
        this.preAdjacentPoints.remove(preAdjacentPoint);
    }

    @Override
    public boolean equals(Object obj) {
        if(this==obj){
            return true;
        }
        if(obj==null){
            return false;
        }
        if(obj instanceof GraphNode){
            GraphNode node = (GraphNode) obj;
            if(this.originalCodeStr.equals(node.getOriginalCodeStr()) && this.codeLineNum==node.codeLineNum){
                return true;
            }
        }
        return false;
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + codeLineNum;
        result = prime * result + ((originalCodeStr == null) ? 0 : originalCodeStr.hashCode());
        return result;
    }

    @Override
    public String toString() {
        return originalCodeStr;
    }


}
