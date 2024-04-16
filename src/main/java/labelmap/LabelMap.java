package irit.labelmap;

import irit.similarity.EmbeddingManager;
import org.apache.jena.rdf.model.Model;
import org.apache.jena.rdf.model.RDFNode;
import org.apache.jena.rdf.model.Statement;
import org.apache.jena.rdf.model.StmtIterator;
import org.apache.jena.riot.RDFDataMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;
import java.util.stream.Collectors;

public class LabelMap {

    private final Map<String, Map<String, Set<String>>> spm = new HashMap<>();
    private final Map<String, Map<String, Set<String>>> spmi = new HashMap<>();
    private final Map<String, Map<String, Set<String>>> pom = new HashMap<>();
    private final Map<String, Map<String, Set<String>>> som = new HashMap<>();
    private final Map<String, String> typeMap = new HashMap<>();

    private List<String> collect;
    private INDArray vstack, vNorm;


    public LabelMap(String path) {
        load(path);
    }

    public void load(String path) {
        Model defaultModel = RDFDataMgr.loadModel(path);


        StmtIterator stmtIterator = defaultModel.listStatements();

        while (stmtIterator.hasNext()) {
            Statement statement = stmtIterator.nextStatement();
            String s = statement.getSubject().toString();
            String p = statement.getPredicate().toString();
            String o = statement.getObject().toString();

            typeMap.put(s, getType(statement.getSubject()));
            typeMap.put(p, getType(statement.getPredicate()));
            typeMap.put(o, getType(statement.getObject()));


            String si = s.toLowerCase();
            String pi = p.toLowerCase();
            String oi = o.toLowerCase();


            method1(s, p, o);


            method2(s, p, o, si, pi, oi);

        }


    }

    private void method2(String s, String p, String o, String si, String pi, String oi) {
        method3(s, p, o, si, pi, oi, spmi, som, spm);
    }

    private void method3(String s, String p, String o, String si, String pi, String oi, Map<String, Map<String, Set<String>>> spmi, Map<String, Map<String, Set<String>>> som, Map<String, Map<String, Set<String>>> spm) {
        if (!spmi.containsKey(oi)) spmi.put(oi, new HashMap<>());
        if (!spmi.get(oi).containsKey(pi)) spmi.get(oi).put(pi, new HashSet<>());

        spmi.get(oi).get(pi).add(si);

        if (!som.containsKey(p)) spm.put(p, new HashMap<>());
        if (!spm.get(p).containsKey(s)) spm.get(p).put(s, new HashSet<>());

        spm.get(p).get(s).add(o);
    }

    private void method1(String s, String p, String o) {
        method3(p, s, o, s, p, o, spm, pom, pom);
    }

    public Set<String> getSimilar(String v) {
        v = v.toLowerCase();
        Set<String> result = new HashSet<>();

        if (spmi.containsKey(v)) {
            Map<String, Set<String>> stringSetMap = spmi.get(v);

            for (Set<String> value : stringSetMap.values()) {
                result.addAll(value);
            }


            if (stringSetMap.containsKey("http://www.w3.org/2008/05/skos-xl#literalForm")) {
                Set<String> strings = stringSetMap.get("http://www.w3.org/2008/05/skos-xl#literalForm");
                for (String string : strings) {
                    Map<String, Set<String>> stringSetMap1 = spmi.get(string);
                    if (stringSetMap1.containsKey("http://www.w3.org/2008/05/skos-xl#prefLabel")) {
                        Set<String> strings1 = stringSetMap1.get("http://www.w3.org/2008/05/skos-xl#prefLabel");
                        result.addAll(strings1);
                    }
                }
            }
        }

        return result;
    }

    private boolean isBNode(String v) {

        float digitCount = 0;

        for (int i = 0; i < v.length(); i++) {
            if (Character.isDigit(v.charAt(i))) digitCount++;
        }

        float digitProportion = digitCount / v.length();
        return digitProportion > 0.30f;
    }
    public Set<String> getSimilarEmb(String v, float embThreshold) {
        v = v.toLowerCase();
        Set<String> result = new HashSet<>();


        if(collect == null){
            collect = spmi.keySet().stream().filter(s -> !isBNode(s)).collect(Collectors.toList());
            vstack = Nd4j.vstack(collect.stream().map(EmbeddingManager::get).toList()).detach();
            vNorm = vstack.norm2(1).detach();
        }

        Map.Entry<Integer, Float> integerFloatEntry = EmbeddingManager.maxArgSim(v, vstack, vNorm);

        if (integerFloatEntry.getValue() > embThreshold) {
            Map<String, Set<String>> stringSetMap = spmi.get(collect.get(integerFloatEntry.getKey()));
            for (Set<String> value : stringSetMap.values()) {
                result.addAll(value);
            }
        }


        Nd4j.getMemoryManager().invokeGc();

        return result;
    }


    public Set<String> getMatched(String v) {

        Set<String> result = new HashSet<>();

        method4(v, result, pom);
        method4(v, result, spm);

        return result;
    }

    private void method4(String v, Set<String> result, Map<String, Map<String, Set<String>>> pom) {
        result.addAll(pom.getOrDefault(v, Map.of()).getOrDefault("http://www.w3.org/2000/01/rdf-schema#seeAlso", Set.of()));
        result.addAll(pom.getOrDefault(v, Map.of()).getOrDefault("http://www.w3.org/2002/07/owl#sameAs", Set.of()));
        result.addAll(pom.getOrDefault(v, Map.of()).getOrDefault("http://www.w3.org/2004/02/skos/core#closeMatch", Set.of()));
        result.addAll(pom.getOrDefault(v, Map.of()).getOrDefault("http://www.w3.org/2004/02/skos/core#exactMacth", Set.of()));
    }


    public boolean exists(String v) {
        Set<String> excludedProperties = Set.of(
                "http://www.w3.org/2002/07/owl#sameAs",
                "http://www.w3.org/2004/02/skos/core#closeMatch",
                "http://www.w3.org/2004/02/skos/core#exactMatch"
        );
        int sc = 0;

        for (String s : pom.getOrDefault(v, Map.of()).keySet()) {
            if (!excludedProperties.contains(s)) sc++;
        }

        int oc = 0;
        for (String s : spm.getOrDefault(v, Map.of()).keySet()) {
            if (!excludedProperties.contains(s)) oc++;
        }
        return sc + oc > 0;
    }


    public Set<String> labels(String v) {
        Set<String> result = new HashSet<>();
        for (Set<String> value : pom.getOrDefault(v, Map.of()).values()) {
            result.addAll(value);
        }

        Set<String> orDefault = pom.getOrDefault(v, Map.of()).getOrDefault("http://www.w3.org/2008/05/skos-xl#prefLabel", Set.of());
        for (String s : orDefault) {
            Set<String> orDefault1 = pom.getOrDefault(s, Map.of()).getOrDefault("http://www.w3.org/2008/05/skos-xl#literalForm", Set.of());
            result.addAll(orDefault1);
        }

        return result.stream().filter(s -> typeMap.get(s).equals("Literal")).collect(Collectors.toSet());
    }


    public String getType(RDFNode r) {
        if (r.isLiteral()) return "Literal";
        if (r.isAnon()) return "Anon";
        if (r.isResource()) return "Resource";
        if (r.isURIResource()) return "URIResource";
        return "None";
    }


    public Set<String> types(String value) {
//        SELECT DISTINCT ?type WHERE {" +
//            value + " a ?type."
//                    + "filter(isIRI(?type))}

        Set<String> orDefault = pom.getOrDefault(value, Map.of()).getOrDefault("http://www.w3.org/1999/02/22-rdf-syntax-ns#type", Set.of());

        return orDefault.stream().filter(s -> typeMap.get(s).equals("URIResource") || typeMap.get(s).equals("Resource")).collect(Collectors.toSet());
    }


    public List<Map<String, String>> pathBetween(String v1, String v2, int maxDepth) {
        if (v1.equals(v2)) return List.of();
        Set<String> visited = new HashSet<>();
        Queue<Tree<String>> queue = new LinkedList<>();
        queue.add(new Tree<>(v1, null, 0));
        while (!queue.isEmpty()) {
            Tree<String> node = queue.poll();
            if (visited.contains(node.getValue())) continue;
            visited.add(node.getValue());
            if (node.getValue().equals(v2)) {
                List<String> result = new ArrayList<>();
                while (node != null) {
                    result.add(node.getValue());
                    node = node.getParent();
                }

                Map<String, String> map = new HashMap<>();
                map.put("v1", result.removeLast());
                map.put("v2", result.removeFirst());

                for (int i = 0, j = 1; i < result.size(); i++) {
                    if (i % 2 == 1) continue;
                    map.put("p" + j, result.get(i));
                    j++;
                }

                return List.of(map);
            }
            for (Map.Entry<String, Set<String>> stringSetEntry : pom.getOrDefault(node.getValue(), Map.of()).entrySet()) {
                if (node.getDepth() >= maxDepth) continue;
                Tree<String> stringTree = new Tree<>(stringSetEntry.getKey(), node, node.getDepth() + 1);
                for (String s : stringSetEntry.getValue()) {
                    Tree<String> child = new Tree<>(s, stringTree, stringTree.getDepth() + 1);
                    stringTree.addChild(child);
                    queue.add(child);
                }
            }

        }

        return List.of();
    }


}
