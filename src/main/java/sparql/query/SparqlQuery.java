package irit.sparql.query;

import irit.resource.IRI;

import java.util.*;
import java.util.Map.Entry;
import java.util.regex.Matcher;
import java.util.regex.Pattern;


public abstract class SparqlQuery {
    protected final Map<String, IRI> iriList;
    private final Set<Entry<String, String>> prefix;
    protected String where;
    protected String mainQuery;

    public SparqlQuery(String query) {
        prefix = new HashSet<>();
        iriList = new HashMap<>();
        addDefaultPrefixes();
        retrievePrefixes(query);
        where = "";
        retrieveIRIs();
    }

    public void addDefaultPrefixes() {
        prefix.add(new AbstractMap.SimpleEntry<>("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#"));
        prefix.add(new AbstractMap.SimpleEntry<>("rdfs", "http://www.w3.org/2000/01/rdf-schema#"));
        prefix.add(new AbstractMap.SimpleEntry<>("owl", "http://www.w3.org/2002/07/owl#"));
        prefix.add(new AbstractMap.SimpleEntry<>("xsd", "http://www.w3.org/2001/XMLSchema#"));
        prefix.add(new AbstractMap.SimpleEntry<>("skos", "http://www.w3.org/2004/02/skos/core#"));
        prefix.add(new AbstractMap.SimpleEntry<>("skos-xl", "http://www.w3.org/2008/05/skos-xl#"));
    }

    public void retrievePrefixes(String aQuery) {
        aQuery = aQuery.trim().replaceAll("PREFIX", "prefix");
        mainQuery = "";

        if (aQuery.contains("prefix")) {
            String[] pref = aQuery.split("prefix");
            for (int j = 0; j < pref.length; j++) {
                String str;
                if (!pref[0].isEmpty())
                    str = pref[0];
                else
                    str = pref[pref.length - 1];
                mainQuery = str.substring(str.indexOf('>') + 1);
            }

            for (String s : pref) {
                String currPrefix = s.trim();
                if (!currPrefix.isEmpty() && currPrefix.indexOf('<') != -1 && currPrefix.indexOf('>') != -1) {
                    int begin = currPrefix.indexOf('<');
                    int end = currPrefix.indexOf('>');
                    String ns = currPrefix.substring(0, currPrefix.indexOf(':')).trim();
                    String iri = currPrefix.substring(begin + 1, end).trim();
                    prefix.add(new AbstractMap.SimpleEntry<>(ns, iri));
                    mainQuery = Pattern.compile(ns + ":([A-Za-z\\d_-]+)").matcher(mainQuery).replaceAll("<" + iri + "$1>");
                }
            }
        } else {
            mainQuery = aQuery;
        }
    }

    public void retrieveIRIs() {
        Pattern patternIRI = Pattern.compile("<[^>]+>");
        Matcher matcherIRI = patternIRI.matcher(mainQuery);
        while (matcherIRI.find()) {
            if (!matcherIRI.group().equals("<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")) {
                IRI iri = new IRI(matcherIRI.group());
                iriList.put(matcherIRI.group(), iri);
            }
        }

        for (Entry<String, String> m : prefix) {
            if (m.getKey() != null) {
                Pattern patt = Pattern.compile("<" + m.getValue() + "([^>]+)>");
                Matcher match = patt.matcher(mainQuery);
                while (match.find()) {
                    if (!match.group().equals("<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")) {
                        iriList.get(match.group()).addLabel(match.group(1));
                    }
                }

            }
        }

    }

    public Map<String, IRI> getIRIList() {
        return iriList;
    }

    public String toUnchangedString() {
        return mainQuery;
    }
}
