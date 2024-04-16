package irit.complex.answer;

import irit.complex.subgraphs.InstantiatedSubgraph;
import irit.complex.subgraphs.Path;
import irit.main.RunArgs;
import irit.resource.IRI;
import irit.resource.Resource;
import irit.similarity.EmbeddingManager;
import irit.sparql.query.select.SparqlSelect;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class PairAnswer extends Answer {
    final Resource r1;
    final Resource r2;
    boolean similarLooked;

    public PairAnswer(Resource r1, Resource r2) {
        if (r1.isIRI()) {
            this.r1 = new IRI("<" + r1 + ">");
        } else {
            this.r1 = r1;
        }
        if (r2.isIRI()) {
            this.r2 = new IRI("<" + r2 + ">");
        } else {
            this.r2 = r2;
        }
        similarLooked = false;
    }

    public void retrieveIRILabels(String endpointURL) {
        if (r1 instanceof IRI) {
            ((IRI) r1).retrieveLabels(endpointURL);
        }
        if (r2 instanceof IRI) {
            ((IRI) r2).retrieveLabels(endpointURL);
        }

    }


    public void getSimilarIRIs(String targetEndpoint) {
        if (!similarLooked) {

            if (r1 instanceof IRI iri) {
                iri.findSimilarResource(targetEndpoint);

            }
            if (r2 instanceof IRI iri) {
                iri.findSimilarResource(targetEndpoint);

            }

            similarLooked = true;
        }
    }

    public void getSimilarIRIsEmb(String targetEndpoint, float embThreshold) {
        if (!similarLooked) {

            if (r1 instanceof IRI iri) {
                iri.findSimilarResourceEmb(targetEndpoint, embThreshold);

            }
            if (r2 instanceof IRI iri) {
                iri.findSimilarResourceEmb(targetEndpoint, embThreshold);

            }

            similarLooked = true;
        }
    }

    public void getExistingMatches(String sourceEndpoint, String targetEndpoint) {

        if (r1 instanceof IRI iri) {
            iri.findExistingMatches(sourceEndpoint, targetEndpoint);
        }
        if (r2 instanceof IRI iri) {
            iri.findExistingMatches(sourceEndpoint, targetEndpoint);
        }

    }


    public Set<InstantiatedSubgraph> findCorresponding(Set<String> queryLabels, String targetEndpoint, double similarityThreshold, int currentLen, int maxLen) {

        Set<InstantiatedSubgraph> paths = new HashSet<>();

        if (currentLen > maxLen) return paths;


        if (hasR1Match() && hasR2Match()) {
            for (Resource x : r1.getSimilarIRIs()) {
                for (Resource y : r2.getSimilarIRIs()) {
                    queryPaths(targetEndpoint, paths, queryLabels, x, y);

                }
            }
        } else if (hasR1Match()) {
            if (!r2.isIRI()) {
                for (IRI x : r1.getSimilarIRIs()) {
                    queryPaths(targetEndpoint, paths, queryLabels, x, r2);
                }
            }
        } else if (hasR2Match()) {
            if (!r1.isIRI()) {
                for (IRI y : r2.getSimilarIRIs()) {
                    queryPaths(targetEndpoint, paths, queryLabels, r1, y);
                }
            }
        }


        for (InstantiatedSubgraph p : paths) {
            if (p instanceof Path pt) {

                pt.compareLabel(queryLabels, similarityThreshold, targetEndpoint, 0.5);

            } else {
                System.err.println("problem in Pair answer: instantiated subgraph is not a path...");
            }
        }


        if (paths.isEmpty() && !similarLooked) {
            getSimilarIRIs(targetEndpoint);
            System.out.println("No path found, similar answers : " + printMatchedEquivalents());
            paths = findCorresponding(queryLabels, targetEndpoint, similarityThreshold, currentLen + 1, maxLen);
        }

        return paths;
    }

    public Set<InstantiatedSubgraph> findCorresponding(INDArray queryLabels, String targetEndpoint, double similarityThreshold, int currentLen, int maxLen) {

        Set<InstantiatedSubgraph> paths = new HashSet<>();

        if (currentLen > maxLen) return paths;


        if (hasR1Match() && hasR2Match()) {
            for (Resource x : r1.getSimilarIRIs()) {
                for (Resource y : r2.getSimilarIRIs()) {
                    queryPaths(targetEndpoint, paths, queryLabels, x, y);

                }
            }
        } else if (hasR1Match()) {
            if (!r2.isIRI()) {
                for (IRI x : r1.getSimilarIRIs()) {
                    queryPaths(targetEndpoint, paths, queryLabels, x, r2);
                }
            }
        } else if (hasR2Match()) {
            if (!r1.isIRI()) {
                for (IRI y : r2.getSimilarIRIs()) {
                    queryPaths(targetEndpoint, paths, queryLabels, r1, y);
                }
            }
        }


        for (InstantiatedSubgraph p : paths) {
            if (p instanceof Path pt) {

                pt.compareLabel(queryLabels, similarityThreshold, targetEndpoint, 0.5);

            } else {
                System.err.println("problem in Pair answer: instantiated subgraph is not a path...");
            }
        }


        if (paths.isEmpty() && !similarLooked) {
            getSimilarIRIs(targetEndpoint);
            System.out.println("No path found, similar answers : " + printMatchedEquivalents());
            paths = findCorresponding(queryLabels, targetEndpoint, similarityThreshold, currentLen + 1, maxLen);
        }

        return paths;
    }

    public Set<InstantiatedSubgraph> findCorrespondingEmb(INDArray queryLabels, String targetEndpoint, double similarityThreshold, int currentLen, int maxLen) {

        Set<InstantiatedSubgraph> paths = new HashSet<>();

        if (currentLen > maxLen) return paths;


        if (hasR1Match() && hasR2Match()) {
            for (Resource x : r1.getSimilarIRIs()) {
                for (Resource y : r2.getSimilarIRIs()) {
                    queryPaths(targetEndpoint, paths, queryLabels, x, y);

                }
            }
        } else if (hasR1Match()) {
            if (!r2.isIRI()) {
                for (IRI x : r1.getSimilarIRIs()) {
                    queryPaths(targetEndpoint, paths, queryLabels, x, r2);
                }
            }
        } else if (hasR2Match()) {
            if (!r1.isIRI()) {
                for (IRI y : r2.getSimilarIRIs()) {
                    queryPaths(targetEndpoint, paths, queryLabels, r1, y);
                }
            }
        }


        for (InstantiatedSubgraph p : paths) {
            if (p instanceof Path pt) {

                pt.compareLabel(queryLabels, similarityThreshold, targetEndpoint, 0.5);

            } else {
                System.err.println("problem in Pair answer: instantiated subgraph is not a path...");
            }
        }


        if (paths.isEmpty() && !similarLooked) {
            getSimilarIRIs(targetEndpoint);
            System.out.println("No path found, similar answers : " + printMatchedEquivalents());
            paths = findCorresponding(queryLabels, targetEndpoint, similarityThreshold, currentLen + 1, maxLen);
        }

        return paths;
    }

    public void queryPaths(String targetEndpoint, Set<InstantiatedSubgraph> paths, Set<String> queryLabels, Resource x, Resource y) {

        Path p = new Path(x, y, targetEndpoint, 5);
        if (!p.pathFound()) return;

        p.getMostSimilarTypes(targetEndpoint, queryLabels, 0.0);
        paths.add(p);
    }

    public void queryPaths(String targetEndpoint, Set<InstantiatedSubgraph> paths, INDArray queryLabels, Resource x, Resource y) {

        Path p = new Path(x, y, targetEndpoint, 5);
        if (!p.pathFound()) return;

        p.getMostSimilarTypes(targetEndpoint, queryLabels, 0.0);
        paths.add(p);
    }

    public Set<InstantiatedSubgraph> findCorrespondingSubGraph(SparqlSelect query, RunArgs runArgs, double similarityThreshold) {
        final Set<String> queryLabels = query.getLabels();
        INDArray cqaEmb = null;

        if (runArgs.getSimType().equals("cqa_emb")) {
            List<INDArray> cqaLabelsEmbs = queryLabels.stream().map(EmbeddingManager::get).toList();
            cqaEmb = Nd4j.vstack(cqaLabelsEmbs).mean(0);
            return findCorresponding(cqaEmb, runArgs.getTargetName(), similarityThreshold, 0, 5);

        } else if (runArgs.getSimType().equals("sub_emb")) {
            List<INDArray> cqaLabelsEmbs = queryLabels.stream().map(EmbeddingManager::get).toList();
            cqaEmb = Nd4j.vstack(cqaLabelsEmbs).mean(0);
            return findCorrespondingEmb(cqaEmb, runArgs.getTargetName(), similarityThreshold, 0, 5);
        } else {
            return findCorresponding(queryLabels, runArgs.getTargetName(), similarityThreshold, 0, 5);

        }


    }

    public boolean hasMatch() {
        boolean match = !r1.isIRI() || hasR1Match();
        if (r2.isIRI() && !hasR2Match()) {
            match = false;
        }
        return match && hasR1Match() && hasR2Match();
    }

    private boolean hasR1Match() {
        return !r1.getSimilarIRIs().isEmpty();
    }

    private boolean hasR2Match() {
        return !r2.getSimilarIRIs().isEmpty();
    }

    public String toString() {
        return r1.toString() + " " + r2.toString();
    }

    public String printMatchedEquivalents() {
        return r1.getSimilarIRIs().toString() + " <--> " + r2.getSimilarIRIs().toString();
    }

}
