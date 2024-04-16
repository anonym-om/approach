package irit.resource;

import irit.complex.answer.QueryTemplate;
import irit.dataset.DatasetManager;
import irit.sparql.SparqlProxy;
import org.apache.jena.rdf.model.RDFNode;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class Resource {

    protected final String value;
    public final HashSet<IRI> similarIRIs;
    private final Pattern pattern = Pattern.compile("[a-z][/:#]");
    private static final Pattern inp = Pattern.compile("\\\\");


    public Resource(String val) {
        value = Resource.inp.matcher(val).replaceAll("");
        similarIRIs = new HashSet<>();
    }

    public boolean isIRI() {
        Matcher matcher = pattern.matcher(value);
        return !value.contains(" ") && matcher.find();
    }

    static final QueryTemplate similar = new QueryTemplate("""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX skos-xl: <http://www.w3.org/2008/05/skos-xl#>
            PREFIX owl: <http://www.w3.org/2002/07/owl#>

            SELECT DISTINCT ?x WHERE {
            {?x ?z ?label.}
            UNION {?x skos-xl:prefLabel ?z.
            ?z skos-xl:literalForm ?label.}
            filter (regex(?label, "^{{labelValue}}$","i")).
            }""");


    public void findSimilarResource(String targetEndpoint) throws Exception {
        Map<String, String> substitution = new HashMap<>();
        substitution.put("labelValue", value);
        if (value.length() > 1) {
            substitution.put("LabelValue", value.substring(0, 1).toUpperCase() + value.substring(1));
        } else {
            substitution.put("LabelValue", "\"" + value.toUpperCase() + "\"");
        }

        querySimilarIri(substitution, targetEndpoint);

        substitution.put("labelValue", "\"" + value.substring(0, 1).toUpperCase() + value.substring(1) + "\"@en");
        querySimilarIri(substitution, targetEndpoint);

        substitution.put("labelValue", "\"" + value.substring(0, 1).toUpperCase() + value.substring(1) + "\"");
        querySimilarIri(substitution, targetEndpoint);

    }

    public void findSimilarResourceEmb(String targetEndpoint, float embThreshold) {
        System.out.println("Emb");
    }

    private void querySimilarIri(Map<String, String> substitution, String targetEndpoint) throws Exception {
        String query = Resource.similar.substitute(substitution);

        List<Map<String, RDFNode>> ret = SparqlProxy.query(targetEndpoint, query);

        for (Map<String, RDFNode> stringRDFNodeMap : ret) {
            String s = stringRDFNodeMap.get("x").toString().replaceAll("\"", "");
            similarIRIs.add(new IRI("<" + s + ">"));
        }
    }

    public String toValueString() {
        if (!(isIRI())) {
            return "\"" + value + "\"";
        } else {
            return toString();
        }
    }

    public String toString() {
        return value;
    }

    public int hashCode() {
        return value.hashCode();
    }

    public boolean equals(Object obj) {
        if (obj instanceof Resource) {
            return value.equals(((Resource) obj).value);
        } else {
            return false;
        }
    }

    public HashSet<IRI> getSimilarIRIs() {
        return similarIRIs;
    }


    public String getValue() {
        return value;
    }
}
