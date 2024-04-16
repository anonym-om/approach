package irit.dataset;

import irit.labelmap.LabelMap;
import org.apache.jena.query.Dataset;
import org.apache.jena.query.DatasetFactory;
import org.apache.jena.rdf.model.Model;
import org.apache.jena.rdf.model.ModelFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

public class DatasetManager {

    private static DatasetManager instance;
    public final Map<String, LabelMap> labelMaps;
    private final Map<String, Dataset> datasetMap;

    private DatasetManager() {
        datasetMap = new HashMap<>();
        labelMaps = new HashMap<>();
    }

    public static DatasetManager getInstance() {
        if (instance == null) instance = new DatasetManager();
        return instance;
    }


    public void load(String name, String path) throws IOException {
        Map<String, String> typeMap = Map.of("owl", "RDF/XML", "rdf", "RDF/XML", "ttl", "TURTLE", "nt", "NTRIPLES");
        Model m = ModelFactory.createDefaultModel();
        m.read(Files.newInputStream(Paths.get(path)), null, typeMap.get(path.substring(path.lastIndexOf(".") + 1)));
        Dataset dataset = DatasetFactory.create(m);
        datasetMap.put(name, dataset);
        labelMaps.put(name, new LabelMap(path));
    }

    public void close() {
        for (Dataset value : datasetMap.values()) {
            value.close();
        }
        labelMaps.clear();
    }

    public Dataset get(String name) {
        return datasetMap.get(name);
    }


}
