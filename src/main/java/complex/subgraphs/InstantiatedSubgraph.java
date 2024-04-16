package irit.complex.subgraphs;

public class InstantiatedSubgraph implements Comparable<InstantiatedSubgraph> {

    public double getSimilarity() {
        return 0;
    }

    @Override
    public int compareTo(InstantiatedSubgraph s) {

        return Double.compare(s.getSimilarity(), getSimilarity());
    }

}

	
