package irit.sparql.query.select;

import irit.resource.IRI;
import irit.sparql.query.SparqlQuery;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;


public class SparqlSelect extends SparqlQuery {
    private final List<String> selectFocus;
    private static final Pattern pattern = Pattern.compile("""
                select[ \t
                distncDISTNC]+(\\?[A-Za-z\\d_-]+)[ \t
                ]+(\\?*[A-Za-z\\d_-]*[ \t
                ]*)where[ \t
                ]*\\{(.+)}[ \t
                ]*$""");

    public SparqlSelect(String query) {
        super(query);
        mainQuery = mainQuery.trim().replaceAll("SELECT", "select").replaceAll("WHERE", "where").replaceAll("\n", " ");
        selectFocus = new ArrayList<>();

        Matcher matcher = SparqlSelect.pattern.matcher(mainQuery);
        while (matcher.find()) {
            selectFocus.add(matcher.group(1).trim());
            if (!matcher.group(2).trim().isEmpty()) {
                selectFocus.add(matcher.group(2).trim());
            }
            where = matcher.group(3).trim();
        }

    }


    public static List<SparqlSelect> load(String path) throws IOException {
        List<SparqlSelect> sparqlSelects;
        try (var walk = Files.walk(Paths.get(path), 1)) {
            sparqlSelects = walk
                    .filter(path1 -> !Files.isDirectory(path1))
                    .map(path1 -> {
                        try {
                            return Files.readString(path1);
                        } catch (IOException e) {
                            throw new RuntimeException(e);
                        }
                    })
                    .map(SparqlSelect::new).toList();

        }
        return sparqlSelects;

    }

    public String toString() {
        return mainQuery;
    }

    public String toSubgraphForm() {

        String ret = where;
        if (selectFocus.size() > 1) {
            int i = 0;
            for (String sf : selectFocus) {
                ret = ret.replaceAll(sf.replaceAll("\\?", "\\\\?") + " ", "\\?answer" + i + " ");
                ret = ret.replaceAll(sf.replaceAll("\\?", "\\\\?") + "\\.", "\\?answer" + i + ".");
                ret = ret.replaceAll(sf.replaceAll("\\?", "\\\\?") + "}", "\\?answer" + i + "}");
                ret = ret.replaceAll(sf.replaceAll("\\?", "\\\\?") + "\\)", "\\?answer" + i + ")");
                i++;
            }
        } else {
            ret = ret.replaceAll(selectFocus.getFirst().replaceAll("\\?", "\\\\?") + " ", "?answer ");
            ret = ret.replaceAll(selectFocus.getFirst().replaceAll("\\?", "\\\\?") + "\\.", "?answer.");
            ret = ret.replaceAll(selectFocus.getFirst().replaceAll("\\?", "\\\\?") + "}", "?answer}");
            ret = ret.replaceAll(selectFocus.getFirst().replaceAll("\\?", "\\\\?") + "\\)", "?answer)");
        }
        return ret.replaceAll("\n", " ").replaceAll("\"", "\"");
    }

    public List<String> getSelectFocus() {
        return selectFocus;
    }

    public int getFocusLength() {
        return selectFocus.size();
    }


    public Set<String> getLabels() {
        Set<String> queryLabels = new HashSet<>();

        for (Map.Entry<String, IRI> iri : getIRIList().entrySet()) {
            queryLabels.addAll(iri.getValue().getLabels());
        }
        return queryLabels;
    }
}
