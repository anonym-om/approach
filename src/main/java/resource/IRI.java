package irit.resource;

import irit.complex.subgraphs.Triple;
import irit.dataset.DatasetManager;
import irit.similarity.EmbeddingManager;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class IRI extends Resource {
    public final Set<String> labels;
    private final Set<Triple> triples;
    private final Set<IRI> types;
    private boolean labelsGot;
    private boolean triplesRetrieved;

    public IRI(String iri) {
        super(iri);
        labels = new HashSet<>();
        triples = new HashSet<>();
        types = new HashSet<>();
        labelsGot = false;
        triplesRetrieved = false;
    }


    public void retrieveLabels(String endpointUrl) {
        if (!labelsGot) {
            addLabel(getSuffix());

            String substring = value.substring(1, value.length() - 1);
            Set<String> labels1 = DatasetManager.getInstance().labelMaps.get(endpointUrl).labels(substring);
            for (String item : labels1) {
                String s = item.replaceAll("\"", "").replaceAll("\\^\\^.+", "").toLowerCase();
                Resource res = new Resource(s);
                if (!res.isIRI()) {
                    addLabel(s);
                }

            }
            labelsGot = true;
        }

    }

    public String getSuffix() {
        Pattern pattern = Pattern.compile("<([^>]+)[#/]([A-Za-z0-9_-]+)>");
        Matcher matcher = pattern.matcher(value);
        if (matcher.find()) {
            return matcher.group(2);
        } else {
            return value;
        }
    }

    public void retrieveTypes(String endpointUrl) {

        Set<String> lmTypes = DatasetManager.getInstance().labelMaps.get(endpointUrl).types(value.replaceAll("[<>]", ""));

        for (String s : lmTypes) {
            types.add(new IRI("<" + s + ">"));
        }
        for (IRI type : types) {
            type.retrieveLabels(endpointUrl);
        }
    }

    public IRI findMostSimilarType(String endpointUrl, Collection<String> targetLabels, double threshold) {
        if (getTypes().isEmpty()) {
            retrieveTypes(endpointUrl);
        }
        double scoreTypeMax = -1;
        IRI finalType = null;
        for (IRI type : getTypes()) {
            double scoreType;
            type.retrieveLabels(endpointUrl);
            scoreType = EmbeddingManager.similarity(type.getLabels(), targetLabels, threshold);
            if (scoreTypeMax < scoreType) {
                scoreTypeMax = scoreType;
                finalType = type;
            }
        }
        return finalType;
    }

    public IRI findMostSimilarType(String endpointUrl, INDArray targetLabels, double threshold) {
        if (getTypes().isEmpty()) {
            retrieveTypes(endpointUrl);
        }
        double scoreTypeMax = -1;
        IRI finalType = null;
        for (IRI type : getTypes()) {
            double scoreType;
            type.retrieveLabels(endpointUrl);
            scoreType = EmbeddingManager.similarity(type.getLabels(), targetLabels, threshold);
            if (scoreTypeMax < scoreType) {
                scoreTypeMax = scoreType;
                finalType = type;
            }
        }
        return finalType;
    }

    public void addLabel(String label) {
        labels.add(label.trim());
    }


    public void findExistingMatches(final String sourceEndpoint, final String targetEndpoint) {

        List<IRI> allMatches = new ArrayList<>();

        allMatches.add(this);

        String s1 = value.replaceAll("\\$", "");
        Set<String> matched1 = DatasetManager.getInstance().labelMaps.get(sourceEndpoint).getMatched(s1);
        Set<String> matched2 = DatasetManager.getInstance().labelMaps.get(targetEndpoint).getMatched(s1);

        for (String s : matched1) {
            Resource res = new Resource(s);
            if (res.isIRI()) {
                allMatches.add(new IRI("<" + s + ">"));
            }
        }

        for (String s : matched2) {
            Resource res = new Resource(s);
            if (res.isIRI()) {
                allMatches.add(new IRI("<" + s + ">"));
            }
        }

        for (IRI match : allMatches) {
            if (DatasetManager.getInstance().labelMaps.get(targetEndpoint).exists(match.toString())) {
                similarIRIs.add(match);
            }
        }

    }

    public void findSimilarResource(String targetEndpoint) {
        if (labels.isEmpty()) {
            retrieveLabels(targetEndpoint);
        }
        for (String rawLab : labels) {

            String label = rawLab.replaceAll("[+{}.?^]", "");

            if (label.length() < 1705) {

                Set<String> similar1 = DatasetManager.getInstance().labelMaps.get(targetEndpoint).getSimilar(label);
                for (String stringRDFNodeMap : similar1) {
                    String s = stringRDFNodeMap.replaceAll("\"", "");
                    similarIRIs.add(new IRI("<" + s + ">"));
                }


            }
        }
    }


    public void findSimilarResourceEmb(String targetEndpoint, float embThreshold) {
        if (labels.isEmpty()) {
            retrieveLabels(targetEndpoint);
        }
        for (String rawLab : labels) {

            String label = rawLab.replaceAll("[+{}.?^]", "");

            if (label.length() < 1705) {

                Set<String> similar1 = DatasetManager.getInstance().labelMaps.get(targetEndpoint).getSimilarEmb(label, embThreshold);
                for (String stringRDFNodeMap : similar1) {
                    String s = stringRDFNodeMap.replaceAll("\"", "");
                    similarIRIs.add(new IRI("<" + s + ">"));
                }


            }
        }
    }


    public Set<String> getLabels() {
        return labels;
    }

    public Set<Triple> getTriples() {
        return triples;
    }

    public Set<IRI> getTypes() {
        return types;
    }

    public String toStrippedString() {
        return value.replaceAll("<", "").replaceAll(">", "");
    }


    public boolean isTriplesRetrieved() {
        return triplesRetrieved;
    }

    public void setTriplesRetrieved(boolean triplesRetrieved) {
        this.triplesRetrieved = triplesRetrieved;
    }


}
