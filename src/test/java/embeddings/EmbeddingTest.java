package embeddings;

import irit.similarity.EmbeddingManager;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.file.Files;

public class EmbeddingTest {


    @Test
    public void testRead() throws IOException {

        URL resource = EmbeddingTest.class.getResource("../line.txt");

        String line = Files.readAllLines(new File(resource.getFile()).toPath()).getFirst();


        String[] split = line.split(" ");

        double[] de = new double[split.length];
        for (int j = 0; j < split.length; j++) {
            if(split[j].equals("nan")){
                de[j] = 0;
                continue;
            }
            de[j] = Float.parseFloat(split[j]);
        }

        float[] doubles = EmbeddingManager.floatsFromLine(line);

        Assertions.assertEquals(de.length, doubles.length);

        for (int i = 0; i < de.length; i++) {
            Assertions.assertEquals(de[i], doubles[i], 1e-7);
        }

    }





}
