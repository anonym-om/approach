package irit.complex.subgraphs;

import irit.resource.IRI;
import irit.resource.Resource;
import irit.sparql.SparqlProxy;
import irit.sparql.query.select.SparqlSelect;
import org.apache.jena.rdf.model.RDFNode;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

public class SubgraphForOutput implements Comparable<SubgraphForOutput> {
    double similarity;

    public String toExtensionString() {
        return "";
    }

    public String toIntensionString() {
        return "";
    }

    public String toString() {
        return getAverageSimilarity() + " <-> " + toIntensionString();
    }

    public double getSimilarity() {
        return similarity;
    }

    public double getAverageSimilarity() {
        return similarity;
    }

    public void reassessSimilarityWithCounterExamples(String sourceEndpoint, String targetEndpoint, SparqlSelect sq) {
        double nbTrueExamples = 0;
        double nbCounterExamples = 0;
        double nbRetSource = 0;

        List<Map<String, RDFNode>> resultS = SparqlProxy.query(sourceEndpoint, sq.toString());
        int offset = 0;
        int limit = 10000;
        boolean end = false;
        while (!end) {
            String newQuery = toSPARQLForm();
            newQuery += "\n LIMIT " + limit;
            newQuery += "\n OFFSET " + offset;
            List<Map<String, RDFNode>> resultT = SparqlProxy.query(targetEndpoint, newQuery);
            Iterator<Map<String, RDFNode>> retIterator = resultT.iterator();

            while (retIterator.hasNext() && nbCounterExamples <= 10 * nbRetSource) {
                Map<String, RDFNode> response = retIterator.next();
                if (response.containsKey("answer")) {
                    IRI iriResponse = new IRI("<" + response.get("answer").toString() + ">");
                    iriResponse.findExistingMatches(targetEndpoint, sourceEndpoint);
                    for (IRI sourceRes : iriResponse.getSimilarIRIs()) {
                        if (SparqlProxy.sendAskQuery(sourceEndpoint, "ASK{" + sq.toSubgraphForm().replaceAll("\\?answer", sourceRes.toString()) + "}")) {
                            nbTrueExamples += 1;
                        } else {
                            nbCounterExamples += 1;
                        }
                    }

                }
                if (response.containsKey("answer1")) {
                    Resource r1 = new Resource(response.get("answer0").toString());
                    Resource r2 = new Resource(response.get("answer1").toString());
                    List<Resource> valuesr1Source = new ArrayList<>();
                    List<Resource> valuesr2Source = new ArrayList<>();

                    if (r1.isIRI()) {
                        r1 = new IRI("<" + r1 + ">");
                        ((IRI) r1).findExistingMatches(targetEndpoint, sourceEndpoint);
                        valuesr1Source.addAll(r1.getSimilarIRIs());
                    } else {
                        valuesr1Source.add(r1);
                    }
                    if (r2.isIRI()) {
                        r2 = new IRI("<" + r2 + ">");
                        ((IRI) r2).findExistingMatches(targetEndpoint, sourceEndpoint);
                        valuesr2Source.addAll(r2.getSimilarIRIs());
                    } else {
                        valuesr2Source.add(r2);
                    }
                    for (Resource sourceRes1 : valuesr1Source) {
                        for (Resource sourceRes2 : valuesr2Source) {
                            String query = sq.toSubgraphForm();
                            if (sourceRes1.isIRI()) {
                                query = query.replaceAll("\\?answer0", sourceRes1.toString());
                            } else {
                                query += " Filter(str(?answer0)=" + sourceRes1.toValueString() + ")";
                            }
                            if (sourceRes2.isIRI()) {
                                query = query.replaceAll("\\?answer1", sourceRes2.toString());
                            } else {
                                query += " Filter(str(?answer1)=" + sourceRes2.toValueString() + ")";
                            }
                            query = "ASK{" + query + "}";
                            if (SparqlProxy.sendAskQuery(sourceEndpoint, query)) {
                                nbTrueExamples += 1;
                            } else {
                                nbCounterExamples += 1;
                            }
                        }
                    }
                }
            }
            if (resultS.size() < limit) {
                end = true;
            } else {
                offset += limit;
            }
            if (nbCounterExamples >= 10 * nbRetSource) {
                end = true;
            }
            if (offset > 600000) {
                end = true;
            }
        }

        if (nbTrueExamples + nbCounterExamples == 0) {
            similarity = 0;
        } else {
            double percentageCommonOK = nbTrueExamples / (nbTrueExamples + nbCounterExamples);
            similarity *= percentageCommonOK;
        }

    }

    public String toSPARQLForm() {
        return "";
    }

    @Override
    public int compareTo(SubgraphForOutput o) {
        return Double.compare(getSimilarity(), o.getSimilarity());
    }

}
