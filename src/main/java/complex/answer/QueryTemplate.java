package irit.complex.answer;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class QueryTemplate {
    private final String query;
    private final Set<String> toSubstitute;

    public QueryTemplate(String query) {
        this.query = query;
        toSubstitute = new HashSet<>();
        Pattern p = Pattern.compile("\\{\\{ ?([A-Za-z0-9]+) ?\\}\\}");
        Matcher m = p.matcher(query);
        while (m.find()) {
            toSubstitute.add(m.group(1));
        }
    }


    public String substitute(Map<String, String> substitution) throws Exception {
        String query = this.query;
        if (substitution.keySet().containsAll(toSubstitute)) {
            for (String key : toSubstitute) {
                query = query.replaceAll("\\{\\{ ?" + key + " ?\\}\\}", substitution.get(key));
            }
        } else {
            throw new Exception("Some elements of the substitution " + toSubstitute + "are not resolved by " + substitution);
        }
        return query;
    }
}
