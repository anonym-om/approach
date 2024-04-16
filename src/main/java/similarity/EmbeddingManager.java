package irit.similarity;

import org.apache.commons.text.similarity.LevenshteinDistance;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

public class EmbeddingManager {


    private static final Map<String, INDArray> embs1 = new HashMap<>();
    private static final Pattern pattern = Pattern.compile("([^>]+)[#/]([A-Za-z0-9_-]+)");
    public static long[] embshape;
    private static final LevenshteinDistance levenshteinDistance = LevenshteinDistance.getDefaultInstance();
    private static boolean ignoreCase = false;

    public static double similarity(Set<String> labels1, Collection<String> labels2, double threshold) {
        return EmbeddingManager.embs1.isEmpty() ? stringSimilarity(labels1, labels2, threshold) : embeddingSimilarity(labels1, labels2, threshold);
    }

    public static double similarity(Set<String> labels1, INDArray labels2, double threshold) {
        return embeddingSimilarity(labels1, labels2, threshold);
    }

    public static double similarity(INDArray emb1, INDArray emb2, double threshold) {
        double sim = EmbeddingManager.cosineSim(emb1, emb2);
        return sim < threshold ? 0 : sim;
    }

    public static double similarity(String s1, String s2) {
        INDArray emb1 = EmbeddingManager.get(s1);
        INDArray emb2 = EmbeddingManager.get(s2);
        return EmbeddingManager.cosineSim(emb1, emb2);
    }

    public static Map.Entry<Integer, Float> maxArgSim(String s, INDArray vstack, INDArray vNorm){

        INDArray emb1 = EmbeddingManager.get(s);
        INDArray norm1 = emb1.norm2();


        INDArray dot = emb1.mul(vstack).sum(1);
        INDArray norm = norm1.mul(vNorm);

        for (int i = 0; i < norm.length(); i++) {
            if (norm.getDouble(i) == 0) {
                norm.putScalar(i, 1);
            }
        }

        INDArray sim = dot.div(norm);

        INDArray argMax = sim.argMax();

        int anInt = argMax.getInt(0);
        float aFloat = sim.getFloat(anInt);

        return Map.entry(anInt, aFloat);

    }

    public static double cosineSim(INDArray emb1, INDArray emb2) {
        INDArray dot = emb1.mul(emb2).sum();
        double norm = emb1.norm2Number().doubleValue() * emb2.norm2Number().doubleValue();
        if (norm == 0) return 0;
        return dot.getDouble(0) / norm;
    }


    public static double embeddingSimilarity(Set<String> labels1, Collection<String> labels2, double threshold) {
        double score = 0;

        Set<String> lab1 = labels1.stream().map(s -> s.replaceAll("\n+", " ")).collect(Collectors.toSet());
        Set<String> lab2 = labels2.stream().map(s -> s.replaceAll("\n+", " ")).collect(Collectors.toSet());

        for (String l1 : lab1) {
            for (String l2 : lab2) {

                INDArray emb1 = EmbeddingManager.get(l1);
                INDArray emb2 = EmbeddingManager.get(l2);

                double sim = EmbeddingManager.cosineSim(emb1, emb2);
                sim = sim < threshold ? 0 : sim;
                score += sim;
            }
        }
        return score;
    }

    public static double embeddingSimilarity(Set<String> labels1, INDArray labels2, double threshold) {
        double score = 0;

        Set<String> lab1 = labels1.stream().map(s -> s.replaceAll("\n+", " ")).collect(Collectors.toSet());

        for (String l1 : lab1) {
            INDArray emb1 = EmbeddingManager.get(l1);

            double sim = EmbeddingManager.cosineSim(emb1, labels2);
            sim = sim < threshold ? 0 : sim;
            score += sim;
        }
        return score;
    }

    public static double stringSimilarity(Set<String> labels1, Collection<String> labels2, double threshold) {
        double score = 0;

        Set<String> lab1 = labels1.stream().map(EmbeddingManager::getSuffix).map(String::toLowerCase).collect(Collectors.toSet());
        Set<String> lab2 = labels2.stream().map(EmbeddingManager::getSuffix).map(String::toLowerCase).collect(Collectors.toSet());

        for (String l1 : lab1) {
            for (String l2 : lab2) {

                double sim = 1 - levenshteinDistance.apply(l1, l2) / (float) Math.max(l1.length(), l2.length());
                sim = sim < threshold ? 0 : sim;
                score += sim;

            }
        }
        return score;
    }

    private static String getSuffix(String value) {
        Matcher matcher = pattern.matcher(value);
        return matcher.find() ? matcher.group(2) : value;
    }

    public static void loadEmbs(String n1) throws IOException {
        Scanner scanner = new Scanner(Files.newBufferedReader(Paths.get(n1)));
        int size = Integer.parseInt(scanner.nextLine());
        List<String> keys = new ArrayList<>();

        for (int i = 0; i < size; i++) {
            String line = scanner.nextLine();
            if (line.startsWith("http://") && line.contains("#")) {
                String[] split = line.split("#");
                if (split.length > 1) {
                    line = split[1];
                } else {
                    line = split[0];
                }
            } else {
                line = line.toLowerCase();
            }
            keys.add(line);
        }

        for (String key : keys) {

            INDArray indArray = Nd4j.create(floatsFromLine(scanner.nextLine()));
            if (embshape == null) EmbeddingManager.embshape = indArray.shape();

            if (EmbeddingManager.isIgnoreCase()) key = key.toLowerCase().strip();

            EmbeddingManager.embs1.put(key, indArray);
        }
    }


    public static float[] floatsFromLine(String line) {
        float[] doubles = new float[100];
        int size = 0;

        float pref = 1.0f;
        float value = 0.0f;
        boolean dot = false;
        float currentExp = 0.1f;
        boolean readExp = false;
        float epref = 1.0f;
        float exp = 0.0f;

        boolean readingNan = false;

        for (char c : line.toCharArray()) {
            if (c == '-') {
                if (readExp) {
                    epref = -1.0f;
                } else {
                    pref = -1.0f;
                }
            } else if (Character.isDigit(c)) {
                if (!dot) {
                    value = value * 10f + Character.getNumericValue(c);
                } else {
                    value = value + Character.getNumericValue(c) * currentExp;
                    currentExp *= 0.1f;
                }

                if (readExp) {
                    exp = exp * 10f + Character.getNumericValue(c);
                }
            } else if (c == '.') {
                dot = true;
            } else if (c == ' ') {
                value = value * pref;

                if (readExp) {
                    value = (float) (value * Math.pow(10, exp * epref));
                }


                if (size == doubles.length) {
                    doubles = Arrays.copyOf(doubles, size * 2);
                }

                if (readingNan) {
                    value = 0;
                }

                doubles[size] = value;
                size++;
                pref = 1.0f;
                value = 0.0f;
                dot = false;
                currentExp = 0.1f;
                readExp = false;
                epref = 1.0f;
                exp = 0.0f;
                readingNan = false;
            } else if (c == 'e') {
                readExp = true;

            } else if (c == 'n' || c == 'a') {
                readingNan = true;
            } else {
                throw new RuntimeException("Unknown char " + c);
            }

        }
        value = value * pref;

        if (size == doubles.length) {
            doubles = Arrays.copyOf(doubles, size * 2);
        }

        doubles[size] = value;
        size++;

        return Arrays.copyOf(doubles, size);
    }

    public static void loadEmbeddings(List<String> paths) throws IOException {
        for(String path : paths) {
            loadEmbs(path);
        }
    }



    public static INDArray get(String e1) {
        if (!embs1.containsKey(e1) && EmbeddingManager.isIgnoreCase()) {
            e1 = e1.toLowerCase().strip();
        }

        if (!embs1.containsKey(e1)) {
            return Nd4j.zeros(DataType.FLOAT, embshape);
        }
        return embs1.get(e1);
    }

    public static INDArray embLabels(Collection<String> labels) {
        List<INDArray> cqaLabelsEmbs = labels.stream().map(EmbeddingManager::get).toList();
        return Nd4j.vstack(cqaLabelsEmbs).mean(0);
    }


    public static boolean isIgnoreCase() {
        return ignoreCase;
    }

    public static void setIgnoreCase(boolean ignoreCase) {
        EmbeddingManager.ignoreCase = ignoreCase;
    }
}
