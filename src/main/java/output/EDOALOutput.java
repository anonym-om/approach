package irit.output;

import fr.inrialpes.exmo.align.impl.edoal.*;
import fr.inrialpes.exmo.align.impl.renderer.RDFRendererVisitor;
import fr.inrialpes.exmo.align.parser.SyntaxElement.Constructor;
import fr.inrialpes.exmo.ontowrap.BasicOntology;
import irit.complex.subgraphs.Path;
import irit.complex.subgraphs.PathSubgraph;
import irit.complex.subgraphs.SubgraphForOutput;
import irit.complex.subgraphs.TripleSubgraph;
import irit.complex.utils.SPARQLNode;
import irit.sparql.query.select.SparqlSelect;
import org.semanticweb.owl.align.AlignmentException;

import java.io.IOException;
import java.io.PrintWriter;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;


public class EDOALOutput extends Output {
    private final String outputEDOALfile;
    private EDOALAlignment alignment;

    public EDOALOutput(String source, String target, String file) {
        super(source, target);
        outputEDOALfile = file;
    }

    private static RelationConstruction getConstruction(Matcher matcher12) throws URISyntaxException {
        List<RelationExpression> setPredId = getRelationExpressions(matcher12);
        return new RelationConstruction(Constructor.OR, setPredId);
    }

    private static List<RelationExpression> getRelationExpressions(Matcher matcher12) throws URISyntaxException {
        List<RelationExpression> setPredId = new ArrayList<>();
        String[] preds = matcher12.group(2).trim().split(",");
        for (String predUri : preds) {
            RelationId predInv = new RelationId(new URI(predUri.replaceAll("[<> ]", "")));
            List<RelationExpression> setPredIdInv = new ArrayList<>();
            setPredIdInv.add(predInv);
            RelationConstruction pred = new RelationConstruction(Constructor.INVERSE, setPredIdInv);
            setPredId.add(pred);
        }
        return setPredId;
    }

    private static RelationConstruction getRelationConstruction(Matcher matcher11) throws URISyntaxException {
        List<RelationExpression> setPredId = getExpressions(matcher11);
        return new RelationConstruction(Constructor.OR, setPredId);
    }

    private static List<RelationExpression> getExpressions(Matcher matcher11) throws URISyntaxException {
        List<RelationExpression> setPredId = new ArrayList<>();
        String[] preds = matcher11.group(1).trim().split(",");
        for (String predUri : preds) {
            RelationId pred = new RelationId(new URI(predUri.replaceAll("[<> ]", "")));
            setPredId.add(pred);
        }
        return setPredId;
    }

    public void init() throws AlignmentException, URISyntaxException {
        alignment = new EDOALAlignment();
        BasicOntology<Object> o1 = getBasicOntology(sourceEndpoint);
        BasicOntology<Object> o2 = getBasicOntology(targetEndpoint);

        alignment.init(o1, o2);
    }

    private BasicOntology<Object> getBasicOntology(String sourceEndpoint) throws URISyntaxException {
        BasicOntology<Object> o1 = new BasicOntology<>();
        o1.setURI(new URI(sourceEndpoint));
        o1.setFormalism("owl");
        o1.setFormURI(new URI("http://www.w3.org/TR/owl-guide/"));
        return o1;
    }

    @Override
    public void addToOutput(List<SubgraphForOutput> output, SparqlSelect sq) throws AlignmentException, URISyntaxException {
        Expression sourceExpr = subgraphFormToEDOALEntity(sq.toSubgraphForm(), sq.getSelectFocus());

        for (SubgraphForOutput s : output) {
            if (s.getSimilarity() >= 0) {
                Expression targetExpr = null;
                double score = s.getSimilarity();
                if (score > 1.0) {
                    score = 1.0;
                }
                if (s instanceof TripleSubgraph) {
                    if (s.toIntensionString().contains("somePredicate")) {
                        targetExpr = subgraphFormToEDOALEntity(s.toExtensionString(), sq.getSelectFocus());
                    } else if (s.toIntensionString().contains("someObject") || s.toIntensionString().contains("someSubject")) {
                        if (((TripleSubgraph) s).predicateHasMaxSim()) {
                            targetExpr = subgraphFormToEDOALEntity(s.toIntensionString(), sq.getSelectFocus());
                        } else {
                            targetExpr = subgraphFormToEDOALEntity(s.toExtensionString(), sq.getSelectFocus());
                        }

                    } else {
                        targetExpr = subgraphFormToEDOALEntity(s.toExtensionString(), sq.getSelectFocus());
                    }
                } else if (s instanceof PathSubgraph) {
                    targetExpr = subgraphFormToEDOALProperty(s);
                }


                if (sourceExpr != null & targetExpr != null) {
                    alignment.addAlignCell(sourceExpr, targetExpr, "Equivalence", score);
                }
            }
        }

    }

    public void end() throws IOException, AlignmentException {
        Files.createDirectories(Paths.get(outputEDOALfile).getParent());
        PrintWriter writer = new PrintWriter(outputEDOALfile, StandardCharsets.UTF_8);
        RDFRendererVisitor renderer = new RDFRendererVisitor(writer);
        alignment.render(renderer);
        writer.flush();
        writer.close();
    }

    public Expression subgraphFormToEDOALEntity(String s, List<String> focus) throws URISyntaxException {
        Expression expr = null;
        Iterable<String> subgraphs = setOfUNIONSubgraphs(s);
        Iterable<String> minusSubgraphs = setOfMINUSSubgraphs(s);
        if (focus.size() == 1) {
            ClassExpression theclassExpr;

            List<ClassExpression> classExpr = new ArrayList<>();
            theclassExpr = getClassExpression(subgraphs, classExpr);
            ClassExpression classExprM;
            List<ClassExpression> classExprMSet = new ArrayList<>();
            classExprM = getClassExpression(minusSubgraphs, classExprMSet);

            if (!classExprMSet.isEmpty()) {
                List<ClassExpression> classExprMNotfinal = new ArrayList<>();
                classExprMNotfinal.add(classExprM);
                ClassExpression minusExpr = new ClassConstruction(Constructor.NOT, classExprMNotfinal);
                List<ClassExpression> classExprFinal = new ArrayList<>();
                classExprFinal.add(theclassExpr);
                classExprFinal.add(minusExpr);
                theclassExpr = new ClassConstruction(Constructor.AND, classExprFinal);
            }

            expr = theclassExpr;

        } else if (focus.size() == 2) {
            RelationExpression theRelExpr;
            RelationExpression relM;
            List<RelationExpression> relExprSet = new ArrayList<>();
            theRelExpr = getRelationExpression(subgraphs, relExprSet);
            List<RelationExpression> relExprMSet = new ArrayList<>();
            relM = getRelationExpression(minusSubgraphs, relExprMSet);

            if (!relExprMSet.isEmpty()) {
                List<RelationExpression> relExprMNotfinal = new ArrayList<>();
                relExprMNotfinal.add(relM);
                RelationExpression minusExpr = new RelationConstruction(Constructor.NOT, relExprMNotfinal);
                List<RelationExpression> relExprFinalSet = new ArrayList<>();
                relExprFinalSet.add(theRelExpr);
                relExprFinalSet.add(minusExpr);
                theRelExpr = new RelationConstruction(Constructor.AND, relExprFinalSet);
            }

            expr = theRelExpr;
        } else {
            System.err.println("Trying to find a class expression with more than 1 variable as select");
        }

        return expr;
    }

    private RelationExpression getRelationExpression(Iterable<String> subgraphs, List<RelationExpression> relExprSet) throws URISyntaxException {
        RelationExpression theRelExpr;
        for (String sub : subgraphs) {
            relExprSet.add(subgraphFormToEDOALProperty(sub, "?answer0", "?answer1"));
        }
        if (relExprSet.size() == 1) {
            theRelExpr = relExprSet.getFirst();
        } else {
            theRelExpr = new RelationConstruction(Constructor.OR, relExprSet);
        }
        return theRelExpr;
    }

    private ClassExpression getClassExpression(Iterable<String> subgraphs, List<ClassExpression> classExpr) throws URISyntaxException {
        ClassExpression theclassExpr;
        for (String sub : subgraphs) {
            classExpr.add(subgraphFormToEDOALClass(sub, "?answer"));
        }
        if (classExpr.size() == 1) {
            theclassExpr = classExpr.getFirst();
        } else {
            theclassExpr = new ClassConstruction(Constructor.OR, classExpr);
        }
        return theclassExpr;
    }

    public Iterable<String> setOfUNIONSubgraphs(String s) {
        s = s.replaceAll("[\n\t ]+", " ");
        s = s.replaceAll("[\n\t ]+\\.", ".");
        s = s.replaceAll("\\\\*\\{", "\\\\{");
        s = s.replaceAll("\\\\*}", "\\\\}");
        s = s.replaceAll("minus", "MINUS");
        s = s.replaceAll("MINUS *\\\\\\{([^\\\\}]+)\\\\}", "");
        s = s.replaceAll("union", "UNION");
        List<String> res = new ArrayList<>();
        Pattern pattern1 = Pattern.compile("\\\\\\{([^\\\\}]+)\\\\} *UNION");
        Matcher matcher1 = pattern1.matcher(s);
        if (matcher1.find()) {
            res.add(matcher1.group(1).trim());
            s = s.replaceFirst("\\\\\\{" + matcher1.group(1).replaceAll("\\?", "\\\\?") + "\\\\}", "");
            Pattern pattern2 = Pattern.compile("UNION *\\\\\\{([^\\\\}]+)\\\\}");
            Matcher matcher2 = pattern2.matcher(s);
            while (matcher2.find()) {
                res.add(matcher2.group(1));
                String mgroup = matcher2.group();
                mgroup = mgroup.replaceAll("\\\\*\\{", "\\\\{");
                mgroup = mgroup.replaceAll("\\\\*}", "\\\\}");
                mgroup = mgroup.replaceAll("\\?", "\\\\?");
                mgroup = mgroup.replaceAll("\\.", "\\\\.");

                s = s.replaceAll("\\\\*\\{", "{");
                s = s.replaceAll("\\\\*}", "}");
                s = s.replaceAll(mgroup, "");
            }
            if (!s.trim().isEmpty()) {
                for (int i = 0; i < res.size(); i++) {
                    res.set(i, res.get(i) + s);
                }
            }
        } else {
            res.add(s);
        }
        return res;
    }

    public Iterable<String> setOfMINUSSubgraphs(String s) {
        s = s.replaceAll("[\n\t ]+", " ");
        s = s.replaceAll("[\n\t ]+\\.", ".");
        s = s.replaceAll("\\\\*\\{", "\\\\{");
        s = s.replaceAll("\\\\*}", "\\\\}");
        s = s.replaceAll("minus", "MINUS");
        List<String> res = new ArrayList<>();
        Pattern pattern1 = Pattern.compile("MINUS *\\\\\\{([^\\\\}]+)\\\\}");
        Matcher matcher1 = pattern1.matcher(s);
        while (matcher1.find()) {
            res.add(matcher1.group(1).trim());
        }
        return res;
    }

    //OLD VERSION
    public ClassExpression subgraphFormToEDOALClass(String s, String focus) throws URISyntaxException {
        ClassExpression expr = null;
        s = s.replaceAll("[\n\t ]+", " ");
        s = s.replaceAll("[\n\t ]+\\.", ".");
        s = s.replaceAll("\\\\*\\{", "\\\\{");
        s = s.replaceAll("\\\\*}", "\\\\}");
        focus = focus.replaceAll("\\?", "\\\\?");
        List<ClassExpression> expressions = new ArrayList<>();

        Pattern pattern1 = Pattern.compile(focus + " <([^ ]+)>\\+? (\\?[A-Za-z\\d_-]+)\\.");
        Matcher matcher1 = pattern1.matcher(s);
        while (matcher1.find()) {
            s = s.replaceAll(matcher1.group().replaceAll("\\?", "\\\\?"), "");
            RelationId pred = new RelationId(new URI(matcher1.group(1).trim()));
            ClassExpression newExpression = subgraphFormToEDOALClass(s, matcher1.group(2).trim());

            if (newExpression == null) {
                expressions.add(new ClassOccurenceRestriction(pred, Comparator.GREATER, 0));
            } else {
                expressions.add(new ClassDomainRestriction(pred, false, newExpression));
            }
        }
        Pattern pattern2 = Pattern.compile("(\\?[A-Za-z\\d_-]+) <([^>]+)>\\+? " + focus + "\\.");
        Matcher matcher2 = pattern2.matcher(s);
        while (matcher2.find()) {
            s = s.replaceAll(matcher2.group().replaceAll("\\?", "\\\\?"), "");
            Result result = getResult(s, matcher2);

            if (result.newExpression() == null) {
                expressions.add(new ClassOccurenceRestriction(result.pred(), Comparator.GREATER, 0));
            } else {
                expressions.add(new ClassDomainRestriction(result.pred(), false, result.newExpression()));
            }
        }
        Pattern pattern3 = Pattern.compile(focus + " (a)?(<http://www\\.w3\\.org/1999/02/22-rdf-syntax-ns#type>)? <([^>]+)>\\.");
        Matcher matcher3 = pattern3.matcher(s);
        while (matcher3.find()) {
            s = s.replaceAll(matcher3.group().replaceAll("\\?", "\\\\?"), "");
            ClassId classId = new ClassId(new URI(matcher3.group(3).trim()));
            expressions.add(classId);
        }

        Pattern pattern4 = Pattern.compile(focus + " <([^>]+)>\\+? <([^>]+)>\\.");
        Matcher matcher4 = pattern4.matcher(s);
        while (matcher4.find()) { //It's a CAV (instance)
            s = s.replaceAll(matcher4.group().replaceAll("\\?", "\\\\?"), "");
            RelationId pred = new RelationId(new URI(matcher4.group(1).trim()));
            InstanceId inst = new InstanceId(new URI(matcher4.group(2).trim()));
            expressions.add(new ClassValueRestriction(pred, Comparator.EQUAL, inst));
        }

        Pattern pattern5 = Pattern.compile("<([^>]+)> <([^>]+)>\\+? " + focus + "\\.");
        Matcher matcher5 = pattern5.matcher(s);
        while (matcher5.find()) {
            s = s.replaceAll(matcher5.group().replaceAll("\\?", "\\\\?"), "");
            RelationId predId = new RelationId(new URI(matcher5.group(2).trim()));
            List<RelationExpression> setPredId = new ArrayList<>();
            setPredId.add(predId);
            RelationConstruction pred = new RelationConstruction(Constructor.INVERSE, setPredId);
            InstanceId inst = new InstanceId(new URI(matcher5.group(1).trim()));
            expressions.add(new ClassValueRestriction(pred, Comparator.EQUAL, inst));
        }

        Pattern pattern6 = Pattern.compile(focus + " <([^>]+)>\\+? \"([^\"]+)\"\\.");
        Matcher matcher6 = pattern6.matcher(s);
        while (matcher6.find()) {
            s = s.replaceAll(matcher6.group().replaceAll("\\?", "\\\\?"), "");
            RelationId pred = new RelationId(new URI(matcher6.group(1).trim()));
            Value value = new Value(matcher6.group(2).trim());
            expressions.add(new ClassValueRestriction(pred, Comparator.EQUAL, value));
        }

        Pattern pattern7 = Pattern.compile(focus + " <([^>]+)>\\+? \\\\\\{([^\\\\}]+)\\\\}\\.");
        Matcher matcher7 = pattern7.matcher(s);
        while (matcher7.find()) {
            s = s.replaceAll(matcher7.group().replaceAll("\\?", "\\\\?"), "");
            List<ClassExpression> setOfCAV = new ArrayList<>();
            RelationId pred = new RelationId(new URI(matcher7.group(1).trim()));
            String[] values = matcher7.group(2).trim().split(",");
            for (String v : values) {
                if (v.contains(">") && v.contains(">")) {
                    InstanceId value = new InstanceId(new URI(v.replaceAll("[<> ]", "")));
                    setOfCAV.add(new ClassValueRestriction(pred, Comparator.EQUAL, value));
                } else {
                    Value value = new Value(v);
                    setOfCAV.add(new ClassValueRestriction(pred, Comparator.EQUAL, value));
                }
            }
            expressions.add(new ClassConstruction(Constructor.OR, setOfCAV));
        }

        //SetOfValues predicate focus
        Pattern pattern8 = Pattern.compile("\\\\\\{([^\\\\}]+)\\\\} <([^>]+)>\\+? " + focus + "\\.");
        Matcher matcher8 = pattern8.matcher(s);
        while (matcher8.find()) {
            s = s.replaceAll(matcher8.group().replaceAll("\\?", "\\\\?"), "");
            List<ClassExpression> setOfCAV = new ArrayList<>();
            RelationId predId = new RelationId(new URI(matcher8.group(2).trim()));
            List<RelationExpression> setPredId = new ArrayList<>();
            setPredId.add(predId);
            RelationConstruction pred = new RelationConstruction(Constructor.INVERSE, setPredId);
            String[] values = matcher8.group(1).trim().split(",");
            for (String v : values) {
                InstanceId value = new InstanceId(new URI(v.replaceAll("[<> ]", "")));
                setOfCAV.add(new ClassValueRestriction(pred, Comparator.EQUAL, value));
            }
            expressions.add(new ClassConstruction(Constructor.OR, setOfCAV));
        }


        Pattern pattern9 = Pattern.compile(focus + " \\\\\\{([^\\\\}]+)\\\\} (\\?[A-Za-z\\d_-]+)\\.");
        Matcher matcher9 = pattern9.matcher(s);
        while (matcher9.find()) {
            s = s.replace(matcher9.group(), "");
            Result2 result = getResult2(s, matcher9);
            if (result.newExpression() == null) {
                expressions.add(new ClassOccurenceRestriction(result.relConstr(), Comparator.GREATER, 0));
            } else {
                expressions.add(new ClassDomainRestriction(result.relConstr(), false, result.newExpression()));
            }
        }


        Pattern pattern10 = Pattern.compile("(\\?[A-Za-z\\d_-]+) \\\\\\{([^\\\\}]+)\\\\} " + focus + "\\.");
        Matcher matcher10 = pattern10.matcher(s);
        while (matcher10.find()) {
            s = s.replace(matcher10.group(), "");
            Result3 result = getResult3(s, matcher10);
            if (result.newExpression() == null) {
                expressions.add(new ClassOccurenceRestriction(result.relConstr(), Comparator.GREATER, 0));
            } else {
                expressions.add(new ClassDomainRestriction(result.relConstr(), false, result.newExpression()));
            }
        }


        Pattern pattern11 = Pattern.compile(focus + " \\\\\\{([^\\\\}]+)\\\\} <([^>]+)>\\.");
        Matcher matcher11 = pattern11.matcher(s);
        while (matcher11.find()) {
            s = s.replace(matcher11.group(), "");
            RelationConstruction relConstr = getRelationConstruction(matcher11);
            InstanceId inst = new InstanceId(new URI(matcher11.group(2).trim()));
            expressions.add(new ClassValueRestriction(relConstr, Comparator.EQUAL, inst));
        }


        Pattern pattern12 = Pattern.compile("<([^>]+)> \\\\\\{([^\\\\}]+)\\\\} " + focus + "\\.");
        Matcher matcher12 = pattern12.matcher(s);
        while (matcher12.find()) {
            s = s.replace(matcher12.group(), "");
            RelationConstruction relConstr = getConstruction(matcher12);
            InstanceId inst = new InstanceId(new URI(matcher12.group(1).trim()));
            expressions.add(new ClassValueRestriction(relConstr, Comparator.EQUAL, inst));
        }


        Pattern pattern13 = Pattern.compile(focus + " \\\\\\{([^\\\\}]+)\\\\} \"([^\"]+)\"\\.");
        Matcher matcher13 = pattern13.matcher(s);
        while (matcher13.find()) {
            s = s.replaceAll(matcher13.group().replaceAll("\\?", "\\\\?"), "");
            RelationConstruction relConstr = getRelationConstruction(matcher13);
            Value val = new Value(matcher13.group(2).trim());
            expressions.add(new ClassValueRestriction(relConstr, Comparator.EQUAL, val));
        }


        if (expressions.size() == 1) {
            expr = expressions.getFirst();
        } else if (expressions.size() > 1) {
            expr = new ClassConstruction(Constructor.AND, expressions);
        }
        return expr;
    }

    private Result3 getResult3(String s, Matcher matcher10) throws URISyntaxException {
        List<RelationExpression> setPredId = getRelationExpressions(matcher10);
        ClassExpression newExpression = subgraphFormToEDOALClass(s, matcher10.group(1).trim());
        RelationConstruction relConstr = new RelationConstruction(Constructor.OR, setPredId);
        return new Result3(newExpression, relConstr);
    }

    private Result2 getResult2(String s, Matcher matcher9) throws URISyntaxException {
        List<RelationExpression> setPredId = getExpressions(matcher9);
        ClassExpression newExpression = subgraphFormToEDOALClass(s, matcher9.group(2).trim());
        RelationConstruction relConstr = new RelationConstruction(Constructor.OR, setPredId);
        return new Result2(newExpression, relConstr);
    }

    private Result getResult(String s, Matcher matcher2) throws URISyntaxException {
        RelationId predId = new RelationId(new URI(matcher2.group(2).trim()));
        List<RelationExpression> setPredId = new ArrayList<>();
        setPredId.add(predId);
        RelationConstruction pred = new RelationConstruction(Constructor.INVERSE, setPredId);
        ClassExpression newExpression = subgraphFormToEDOALClass(s, matcher2.group(1).trim());
        return new Result(pred, newExpression);
    }

    public RelationExpression subgraphFormToEDOALProperty(String s, String focus1, String focus2) throws URISyntaxException {
        s = s.replaceAll("[\n\t ]+", " ");
        s = s.replaceAll("[\n\t ]+\\.", ".");
        s = s.replaceAll("\\\\*\\{", "\\\\{");
        s = s.replaceAll("\\\\*}", "\\\\}");
        s = s.replaceAll("\\+", "");
        String sCopy = s;
        Map<String, SPARQLNode> nodes = new HashMap<>();


        Pattern pattern1 = Pattern.compile("(\\?[A-Za-z\\d_-]+) <[^>]+> (\\?[A-Za-z\\d_-]+)\\.");
        Matcher matcher1 = pattern1.matcher(s);
        while (matcher1.find()) {
            s = s.replaceAll(matcher1.group().replaceAll("\\?", "\\\\?"), "");
            String n1 = matcher1.group(1);
            String n2 = matcher1.group(2);
            String triple = matcher1.group();

            SPARQLNode node1 = new SPARQLNode(n1);
            SPARQLNode node2 = new SPARQLNode(n2);
            if (nodes.containsKey(n1)) {
                node1 = nodes.get(n1);
            } else {
                nodes.put(n1, node1);
            }
            if (nodes.containsKey(n2)) {
                node2 = nodes.get(n2);
            } else {
                nodes.put(n2, node2);
            }
            node1.addNeighbour(node2, triple);
            node2.addNeighbour(node1, triple);
        }

        List<SPARQLNode> nodesToVisit = new ArrayList<>();
        nodesToVisit.add(nodes.get(focus1));
        boolean pathFound = false;

        while (!nodesToVisit.isEmpty() && !pathFound && nodesToVisit.getFirst() != null) {
            SPARQLNode currNode = nodesToVisit.getFirst();
            currNode.explore();
            if (currNode.hasNeighbor(focus2)) {
                nodes.get(focus2).setPredecessor(currNode);
                pathFound = true;
                nodesToVisit.removeFirst();
            } else {
                nodesToVisit.removeFirst();
                for (SPARQLNode newNode : currNode.getNeighbors().values()) {
                    if (!newNode.isExplored()) {
                        newNode.setPredecessor(currNode);
                        nodesToVisit.add(newNode);
                    }
                }
            }
        }


        List<String> properties = new ArrayList<>();
        List<Boolean> inverse = new ArrayList<>();
        List<ClassExpression> types = new ArrayList<>();
        if (pathFound) {

            SPARQLNode currNode = nodes.get(focus2);
            String predName = currNode.getPredecessor().getName();
            boolean endOfPath = false;
            while (!endOfPath) {
                String triple = currNode.getTriple(predName);
                Pattern pattern2 = Pattern.compile("(\\?[A-Za-z\\d_-]+) <([^>]+)> (\\?[A-Za-z\\d_-]+)\\.");
                Matcher matcher2 = pattern2.matcher(triple);
                if (matcher2.find()) {
                    sCopy = sCopy.replaceAll(matcher2.group().replaceAll("\\?", "\\\\?"), "");
                    properties.addFirst(matcher2.group(2));

                    if (matcher2.group(1).equals(currNode.getName())) {
                        inverse.addFirst(true);
                    } else {
                        inverse.addFirst(false);
                    }
                }
                types.addFirst(subgraphFormToEDOALClass(sCopy, currNode.getName()));
                if (predName.equals(focus1)) {
                    endOfPath = true;
                } else {
                    currNode = nodes.get(predName);
                    predName = currNode.getPredecessor().getName();
                }
            }
            types.addFirst(subgraphFormToEDOALClass(sCopy, focus1));
        } else {
            types.add(subgraphFormToEDOALClass(s, focus1));
            types.add(subgraphFormToEDOALClass(s, focus2));
        }

        return subgraphFormToEDOALProperty(properties, inverse, types);
    }

    public RelationExpression subgraphFormToEDOALProperty(List<String> properties, List<Boolean> inverse, List<ClassExpression> types) throws URISyntaxException {
        RelationExpression expr;
        List<RelationExpression> setRelComp = new ArrayList<>();
        for (int i = 0; i < inverse.size(); i++) {
            RelationExpression rel;
            List<RelationExpression> setRelAnd = new ArrayList<>();
            if (types.get(i) != null) {
                setRelAnd.add(new RelationDomainRestriction(types.get(i)));
            }
            if (i + 1 == properties.size() && types.get(i + 1) != null) {
                setRelAnd.add(new RelationCoDomainRestriction(types.get(i + 1)));
            }
            if (inverse.get(i)) {
                List<RelationExpression> setRelInv = new ArrayList<>();
                setRelInv.add(new RelationId(new URI(properties.get(i))));
                rel = new RelationConstruction(Constructor.INVERSE, setRelInv);
            } else {
                rel = new RelationId(new URI(properties.get(i)));
            }
            setRelAnd.add(rel);
            if (setRelAnd.size() > 1) {
                setRelComp.add(new RelationConstruction(Constructor.AND, setRelAnd));
            } else {
                setRelComp.add(rel);
            }
        }

        if (properties.isEmpty() && types.size() == 2) {
            List<RelationExpression> setRelAnd = new ArrayList<>();
            if (types.get(0) != null) {
                setRelAnd.add(new RelationDomainRestriction(types.getFirst()));
            }
            if (types.get(1) != null) {
                setRelAnd.add(new RelationCoDomainRestriction(types.get(1)));
            }
            if (setRelAnd.size() > 1) {
                setRelComp.add(new RelationConstruction(Constructor.AND, setRelAnd));
            } else if (setRelAnd.size() == 1) {
                setRelComp.add(setRelAnd.getFirst());
            }


        }


        if (setRelComp.size() == 1) {
            expr = setRelComp.getFirst();
        } else {
            expr = new RelationConstruction(Constructor.COMP, setRelComp);
        }
        return expr;

    }

    public RelationExpression subgraphFormToEDOALProperty(SubgraphForOutput s) throws URISyntaxException {
        List<String> properties = new ArrayList<>();
        List<Boolean> inverse = new ArrayList<>();
        List<ClassExpression> types = new ArrayList<>();
        if (s instanceof PathSubgraph) {
            Path p = ((PathSubgraph) s).getMainPath();
            inverse = p.getInverse();

            for (int i = 0; i < p.getProperties().size(); i++) {
                properties.add(p.getProperties().get(i).toStrippedString());
                if (p.getTypes().get(i) != null) {
                    types.add(new ClassId(new URI(p.getTypes().get(i).toStrippedString())));
                } else {
                    types.add(null);
                }
            }

            if (p.getTypes().get(p.getProperties().size()) != null) {
                types.add(new ClassId(new URI(p.getTypes().get(p.getProperties().size()).toStrippedString())));
            } else {
                types.add(null);
            }
        }
        return subgraphFormToEDOALProperty(properties, inverse, types);
    }

    private record Result3(ClassExpression newExpression, RelationConstruction relConstr) {
    }

    private record Result2(ClassExpression newExpression, RelationConstruction relConstr) {
    }

    private record Result(RelationConstruction pred, ClassExpression newExpression) {
    }
}
