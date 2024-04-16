package irit.output;

import irit.complex.subgraphs.SubgraphForOutput;
import irit.sparql.query.select.SparqlSelect;
import org.semanticweb.owl.align.AlignmentException;

import java.io.IOException;
import java.net.URISyntaxException;
import java.util.List;

public abstract class Output {

    protected final String sourceEndpoint;
    protected final String targetEndpoint;

    public Output(String sourceEndpoint, String targetEndpoint) {
        this.sourceEndpoint = sourceEndpoint;
        this.targetEndpoint = targetEndpoint;

    }

    public void init() throws AlignmentException, URISyntaxException {
    }

    public void addToOutput(List<SubgraphForOutput> output, SparqlSelect sq) throws AlignmentException, URISyntaxException {
    }

    public void end() throws IOException, AlignmentException {
    }

}
