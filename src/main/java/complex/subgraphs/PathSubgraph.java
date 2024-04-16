package irit.complex.subgraphs;

import java.util.ArrayList;

public class PathSubgraph extends SubgraphForOutput {
    final ArrayList<Path> paths;

    public PathSubgraph(Path p) {
        paths = new ArrayList<>();
        paths.add(p);
        similarity = p.getSimilarity();
    }

    public boolean addSubgraph(Path p) {
        boolean added = false;
        if (p.toSubGraphString().equals(paths.getFirst().toSubGraphString())) {
            addSimilarity(p);
            paths.add(p);
            added = true;
        }
        return added;
    }

    public void addSimilarity(Path p) {
        similarity = ((similarity * paths.size()) + p.getSimilarity()) / (paths.size() + 1);
    }

    public String toExtensionString() {
        return paths.getFirst().toSubGraphString();
    }

    public String toSPARQLForm() {
        return "SELECT distinct ?answer0 ?answer1 WHERE {\n" +
                paths.getFirst().toSubGraphString() + "}";
    }

    public String toIntensionString() {
        return paths.getFirst().toSubGraphString();
    }

    public Path getMainPath() {
        return paths.getFirst();
    }


}
