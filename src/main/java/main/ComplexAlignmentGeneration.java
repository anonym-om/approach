package irit.main;

import irit.complex.answer.Answer;
import irit.complex.answer.PairAnswer;
import irit.complex.answer.SingleAnswer;
import irit.complex.subgraphs.*;
import irit.dataset.DatasetManager;
import irit.output.OutputManager;
import irit.resource.IRI;
import irit.resource.Resource;
import irit.similarity.EmbeddingManager;
import irit.sparql.SparqlProxy;
import irit.sparql.query.select.SparqlSelect;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.ArgumentParserException;
import net.sourceforge.argparse4j.inf.Namespace;
import org.apache.jena.rdf.model.RDFNode;
import org.apache.jena.sys.JenaSystem;

import java.util.*;


public class ComplexAlignmentGeneration {


    public static void main(String[] args) {
        JenaSystem.init();
        ArgumentParser parser = RunArgs.buildArgumentParser();

        try {
            Namespace res = parser.parseArgs(args);
            RunArgs runArgs = RunArgs.fromNamespace(res);

            if (runArgs.getSimType().startsWith("i_")) {
                EmbeddingManager.setIgnoreCase(true);
            }

            if (runArgs.getEmbeddings() != null) {
                EmbeddingManager.loadEmbeddings(runArgs.getEmbeddings());
            }

            DatasetManager.getInstance().load(runArgs.getSourceName(), runArgs.getSourceEndpoint());
            DatasetManager.getInstance().load(runArgs.getTargetName(), runArgs.getTargetEndpoint());

            run(runArgs);

            DatasetManager.getInstance().close();


        } catch (ArgumentParserException e) {
            parser.handleError(e);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }


    }


    public static void run(RunArgs runArgs) throws Exception {

        OutputManager outputManager = new OutputManager();
        outputManager.initOutputEdoal(runArgs.getSourceName(), runArgs.getTargetName(), runArgs.getThresholds(), runArgs.getOutputPath());


        for (SparqlSelect sq : runArgs.getQueries()) {
            align(sq, runArgs, outputManager);
        }


        outputManager.endOutput();
    }


    public static void align(SparqlSelect sq, RunArgs runArgs, OutputManager outputManager) throws Exception {
        Set<Answer> matchedAnswers = getMatchedAnswers(runArgs.getLinkType(), sq, runArgs.getSourceName(), runArgs.getTargetName(), runArgs.getMaxMatches(), runArgs.getEmbThreshold());

        for (float threshold : runArgs.getThresholds()) {
            List<SubgraphForOutput> subgraphForOutputs = buildSingleOutput(matchedAnswers, sq, threshold, runArgs);
            if (!subgraphForOutputs.isEmpty()) {
                outputManager.addToOutput(threshold, sq, subgraphForOutputs);
            }

        }

    }


    public static Set<Answer> getMatchedAnswers(String linkType, SparqlSelect sq, String sourceEndpoint, String targetEndpoint, int maxMatches, float embThreshold) throws Exception {
        Map<String, IRI> iriList = sq.getIRIList();
        for (Map.Entry<String, IRI> m : iriList.entrySet()) {
            m.getValue().retrieveLabels(sourceEndpoint);
        }
        List<Answer> answers = new ArrayList<>();
        Set<Answer> matchedAnswers = new HashSet<>();

        getAnswers(sq, sourceEndpoint, targetEndpoint, maxMatches, matchedAnswers, answers);

        answers.sort(Comparator.comparing(Answer::toString));

        ensureAnswers(linkType, sourceEndpoint, targetEndpoint, maxMatches, matchedAnswers, answers, embThreshold);

        return matchedAnswers;
    }

    private static void ensureAnswers(String linkType, String sourceEndpoint, String targetEndpoint, int maxMatches, Set<Answer> matchedAnswers, List<Answer> answers, float embThreshold) throws Exception {
        if (matchedAnswers.isEmpty()) {
            Iterator<Answer> ansIt = answers.iterator();
            while (matchedAnswers.size() < maxMatches && ansIt.hasNext()) {
                Answer ans = ansIt.next();
                ans.retrieveIRILabels(sourceEndpoint);

                if (linkType.equals("emb")) {
                    ans.getSimilarIRIsEmb(targetEndpoint, embThreshold);
                } else {
                    ans.getSimilarIRIs(targetEndpoint);
                }

                if (ans.hasMatch()) {
                    matchedAnswers.add(ans);
                }

            }
        }



    }

    private static void getAnswers(SparqlSelect sq, String sourceEndpoint, String targetEndpoint, int maxMatches, Set<Answer> matchedAnswers, List<Answer> answers) {
        int offsetMatch = 0;

        boolean noMoreSourceAnswers = false;
        int offset = 0;
        int limit = 2000;
        while (!noMoreSourceAnswers && matchedAnswers.size() < maxMatches) {

            String queryLimit = " LIMIT " + limit + "\n OFFSET " + offset;

            if (sq.getFocusLength() == 1) {
                List<Map<String, RDFNode>> ret = loadUnary(sourceEndpoint, sq, answers, queryLimit);
                if (ret.size() < limit) {
                    noMoreSourceAnswers = true;
                }
            } else if (sq.getFocusLength() == 2) {
                List<Map<String, RDFNode>> ret = loadBinary(sourceEndpoint, sq, answers, queryLimit);
                if (ret.size() < limit) {
                    noMoreSourceAnswers = true;
                }
            } else {
                System.out.println("ERROR for query : " + sq.toUnchangedString());
                System.err.println("Problem detected: too many variables in SELECT: can only deal with 1 or 2");
                noMoreSourceAnswers = true;
            }


            if (!noMoreSourceAnswers) {
                offset += limit;
            }

            while (matchedAnswers.size() < maxMatches && offsetMatch < answers.size()) {
                Answer ans = answers.get(offsetMatch);
                ans.getExistingMatches(sourceEndpoint, targetEndpoint);
                if (ans.hasMatch()) {
                    matchedAnswers.add(ans);
                }
                offsetMatch++;
            }
        }
    }


    private static List<SubgraphForOutput> buildSingleOutput(Set<Answer> matchedAnswers, SparqlSelect sq, float threshold, RunArgs runArgs) {
        List<InstantiatedSubgraph> goodSubgraphs = new ArrayList<>();

        List<Answer> answers = new ArrayList<>(matchedAnswers);
        answers.sort(Comparator.comparing(Answer::toString));

        for (Answer ans : answers) {
            Set<InstantiatedSubgraph> localSubgraphs = ans.findCorrespondingSubGraph(sq, runArgs, threshold);
            List<InstantiatedSubgraph> instantiatedSubgraphs = new ArrayList<>(localSubgraphs);
            instantiatedSubgraphs.sort(Comparator.comparing(InstantiatedSubgraph::toString));
            goodSubgraphs.addAll(instantiatedSubgraphs);
        }

        List<SubgraphForOutput> output = getSubgraphForOutputs(goodSubgraphs);


        if (runArgs.isReassess()) {
            for (SubgraphForOutput s : output) {
                s.reassessSimilarityWithCounterExamples(runArgs.getSourceName(), runArgs.getTargetEndpoint(), sq);
            }
        }


        Collections.sort(output);
        output.sort(Comparator.comparing(SubgraphForOutput::toString));

        double simBias = runArgs.getSimType().equals("sub_emb") || runArgs.getSimType().equals("i_sub_emb") ? 0.6 : 0.6;
        return getForOutputs(output, simBias);
    }

    private static List<SubgraphForOutput> getForOutputs(List<SubgraphForOutput> output, double simBias) {
        List<SubgraphForOutput> singleOutput = new ArrayList<>();

        if (!output.isEmpty() && output.getLast().getSimilarity() < simBias && output.getLast().getSimilarity() > 0.01) {
            double sim = output.getLast().getSimilarity();
            boolean moreCorrespondences = true;
            int i = output.size() - 1;
            while (i >= 0 && moreCorrespondences) {
                if (output.get(i).getSimilarity() == sim) {
                    singleOutput.add(output.get(i));
                } else {
                    moreCorrespondences = false;
                }
                i--;
            }
        } else {
            for (SubgraphForOutput s : output) {
                if (s.getSimilarity() >= simBias) {
                    singleOutput.add(s);
                }
            }
        }
        return singleOutput;
    }

    private static List<SubgraphForOutput> getSubgraphForOutputs(List<InstantiatedSubgraph> goodSubgraphs) {
        List<SubgraphForOutput> output = new ArrayList<>();
        for (InstantiatedSubgraph t : goodSubgraphs) {
            boolean added = false;
            Iterator<SubgraphForOutput> it = output.iterator();
            while (it.hasNext() && !added) {
                SubgraphForOutput subG = it.next();
                if (t instanceof Triple tp && subG instanceof TripleSubgraph tg) {
                    added = tg.addSubgraph(tp);
                } else if (t instanceof Path tp && subG instanceof PathSubgraph tg) {
                    added = tg.addSubgraph(tp);
                }
            }
            if (!added) {
                if (t instanceof Triple tp) {
                    output.add(new TripleSubgraph(tp));
                } else if (t instanceof Path tg) {
                    output.add(new PathSubgraph(tg));
                }
            }
        }
        return output;
    }

    private static List<Map<String, RDFNode>> loadBinary(String sourceEndpoint, SparqlSelect sq, List<Answer> answers, String queryLimit) {
        List<Map<String, RDFNode>> result = SparqlProxy.query(sourceEndpoint, sq.toUnchangedString() + queryLimit);
        for (Map<String, RDFNode> response : result) {
            String s1 = response.get(sq.getSelectFocus().get(0).replaceFirst("\\?", "")).toString();
            String s2 = response.get(sq.getSelectFocus().get(1).replaceFirst("\\?", "")).toString();
            boolean type1 = response.get(sq.getSelectFocus().get(0).replaceFirst("\\?", "")).isAnon();
            boolean type2 = response.get(sq.getSelectFocus().get(1).replaceFirst("\\?", "")).isAnon();
            if (!type1 && !type2) {
                if (!s1.isEmpty() && !s2.isEmpty()) {
                    PairAnswer pair = new PairAnswer(new Resource(s1), new Resource(s2));
                    answers.add(pair);
                }
            }
        }
        return result;
    }


    private static List<Map<String, RDFNode>> loadUnary(String sourceEndpoint, SparqlSelect sq, List<Answer> answers, String queryLimit) {
        List<Map<String, RDFNode>> result = SparqlProxy.query(sourceEndpoint, sq.toUnchangedString() + queryLimit);
        for (Map<String, RDFNode> response : result) {
            String s = response.get(sq.getSelectFocus().getFirst().replaceFirst("\\?", "")).toString();
            boolean type = response.get(sq.getSelectFocus().getFirst().replaceFirst("\\?", "")).isAnon();
            if (!type) {
                SingleAnswer singleton = new SingleAnswer(new Resource(s));
                answers.add(singleton);
            }

        }
        return result;
    }



}

