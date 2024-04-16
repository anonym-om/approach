package irit.complex.utils;

import java.util.HashMap;
import java.util.Map;

public class SPARQLNode {

    final String name;
    final HashMap<String, String> triples;
    final HashMap<String, SPARQLNode> neighbors;
    boolean explored;
    SPARQLNode predecessor;

    public SPARQLNode(String n) {
        name = n;
        triples = new HashMap<>();
        neighbors = new HashMap<>();
        explored = false;
    }

    public void addNeighbour(SPARQLNode neighbor, String triple) {
        if (neighbors.containsKey(neighbor.getName())) {
            System.out.println("more than one prop: " + triples.get(neighbor.getName()) + " " + triple);
        } else {
            neighbors.put(neighbor.getName(), neighbor);
            triples.put(neighbor.getName(), triple);
        }
    }

    public Map<String, SPARQLNode> getNeighbors() {
        return neighbors;
    }

    public boolean hasNeighbor(String n) {
        return neighbors.containsKey(n);
    }

    public SPARQLNode getPredecessor() {
        return predecessor;
    }

    public void setPredecessor(SPARQLNode pred) {
        predecessor = pred;
    }

    public boolean isExplored() {
        return explored;
    }

    public void explore() {
        explored = true;
    }

    public String getName() {
        return name;
    }

    public String getTriple(String n) {
        return triples.get(n);
    }

    public String toString() {
        return name + " : " + neighbors.keySet();
    }


}
