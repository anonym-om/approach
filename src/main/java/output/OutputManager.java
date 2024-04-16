package irit.output;

import irit.complex.subgraphs.SubgraphForOutput;
import irit.sparql.query.select.SparqlSelect;
import org.semanticweb.owl.align.AlignmentException;

import java.io.IOException;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class OutputManager {

    private Map<Float, ArrayList<Output>> outputs;

    public void initOutputEdoal(String sourceEndpoint, String targetEndpoint, List<Float> ths, String outputEdoal) throws AlignmentException, URISyntaxException {
        outputs = new HashMap<>();

        for (float th : ths) {
            outputs.put(th, new ArrayList<>());
            String filePath = String.format("%s/%s_%s/th_%s.edoal", outputEdoal, sourceEndpoint, targetEndpoint, String.format("%.1f", th).replaceAll(",", "_"));
            outputs.get(th).add(new EDOALOutput(sourceEndpoint, targetEndpoint, filePath));


            for (Output o : outputs.get(th)) {
                o.init();
            }
        }
    }

    public void addToOutput(float th, SparqlSelect sq, List<SubgraphForOutput> subGraph) throws AlignmentException, URISyntaxException {
        for (Output o : outputs.get(th)) {
            o.addToOutput(subGraph, sq);
        }

    }


    public void endOutput() throws AlignmentException, IOException {
        for (Float th : outputs.keySet()) {
            for (Output o : outputs.get(th)) {
                o.end();
            }
        }

    }


}
