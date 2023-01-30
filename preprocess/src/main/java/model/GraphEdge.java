package model;

import java.util.Objects;

public class GraphEdge {
    
    private EdgeTypes type;
    private int weight;
    private GraphNode originalNode;
    private GraphNode aimNode;

    public GraphEdge(EdgeTypes type, int weight, GraphNode originalNode, GraphNode aimNode) {
        this.type = type;
        this.weight = weight;
        this.originalNode = originalNode;
        this.aimNode = aimNode;
    }

    public GraphEdge(EdgeTypes type, GraphNode originalNode, GraphNode aimNode) {
        this.type = type;
        this.originalNode = originalNode;
        this.aimNode = aimNode;
    }

    public GraphEdge() {
    }

    public EdgeTypes getType() {
        return type;
    }

    public void setType(EdgeTypes type) {
        this.type = type;
    }

    public int getWeight() {
        return weight;
    }

    public void setWeight(int weight) {
        this.weight = weight;
    }

    public GraphNode getOriginalNode() {
        return originalNode;
    }

    public void setOriginalNode(GraphNode originalNode) {
        this.originalNode = originalNode;
    }

    public GraphNode getAimNode() {
        return aimNode;
    }

    public void setAimNode(GraphNode aimNode) {
        this.aimNode = aimNode;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        GraphEdge edge = (GraphEdge) o;
        return type.getColor().equals(edge.type.getColor()) &&
                originalNode.equals(edge.originalNode) &&
                aimNode.equals(edge.aimNode);
    }

    @Override
    public int hashCode() {
        return Objects.hash(type, originalNode, aimNode);
    }

    @Override
    public String toString() {
        return "GraphEdge{" +
                "type=" + type +
                ", weight=" + weight +
                ", originalNode=" + originalNode +
                ", aimNode=" + aimNode +
                '}';
    }
}
