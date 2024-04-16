package irit.complex.subgraphs;

import irit.resource.IRI;
import irit.resource.Resource;
import irit.similarity.EmbeddingManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class Triple extends InstantiatedSubgraph {
    private final IRI subject;
    private final IRI predicate;
    private final Resource object;
    public boolean keepObjectType;
    public boolean keepSubjectType;
    private TripleType tripleType;
    private IRI subjectType;
    private IRI objectType;
    private double objectSimilarity;
    private double subjectSimilarity;
    private double predicateSimilarity;

    public Triple() {
        subject = new IRI("");
        object = new Resource("");
        predicate = new IRI("");
    }

    public Triple(String subject, String predicate, String object, TripleType type) {
        this.subject = new IRI(subject);
        this.predicate = new IRI(predicate);
        Resource r = new Resource(object);
        if (r.isIRI()) {
            this.object = new IRI("<" + object.replaceAll("[<>]", "") + ">");
        } else {
            this.object = r;
        }
        tripleType = type;
        keepObjectType = false;
        keepSubjectType = false;
        objectSimilarity = 0;
        subjectSimilarity = 0;
        predicateSimilarity = 0;
    }

    public void retrieveIRILabels(String targetEndpoint) {
        if (tripleType != TripleType.SUBJECT) {
            subject.retrieveLabels(targetEndpoint);
        }
        if (tripleType != TripleType.PREDICATE) {
            predicate.retrieveLabels(targetEndpoint);
        }
        if (tripleType != TripleType.OBJECT && object instanceof IRI) {
            ((IRI) object).retrieveLabels(targetEndpoint);
        }
    }

    public void retrieveTypes(String targetEndpoint) {
        if (tripleType != TripleType.SUBJECT) {
            subject.retrieveTypes(targetEndpoint);
        }
        if (tripleType != TripleType.PREDICATE) {
            predicate.retrieveTypes(targetEndpoint);
        }
        if (tripleType != TripleType.OBJECT && object instanceof IRI) {
            ((IRI) object).retrieveTypes(targetEndpoint);
        }
    }


    public double compareLabel(Collection<String> targetLabels, double threshold, String targetEndpoint) {
        if (tripleType != TripleType.SUBJECT) {
            subjectType = subject.findMostSimilarType(targetEndpoint, targetLabels, threshold);
            double scoreTypeSubMax = 0;
            if (subjectType != null) {
                scoreTypeSubMax = EmbeddingManager.similarity(subjectType.getLabels(), targetLabels, threshold);
            }
            subjectSimilarity = EmbeddingManager.similarity(subject.getLabels(), targetLabels, threshold);
            if (scoreTypeSubMax > subjectSimilarity) {
                keepSubjectType = true;
                subjectSimilarity = scoreTypeSubMax;
            }
        }
        if (tripleType != TripleType.PREDICATE && !predicate.toString().equals("<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")) {
            predicateSimilarity = EmbeddingManager.similarity(predicate.getLabels(), targetLabels, threshold);
        }
        if (tripleType != TripleType.OBJECT && object instanceof IRI oiri) {
            objectType = oiri.findMostSimilarType(targetEndpoint, targetLabels, threshold);
            if (objectType != null) {
                double scoreTypeObMax = EmbeddingManager.similarity(objectType.getLabels(), targetLabels, threshold);
                objectSimilarity = EmbeddingManager.similarity(oiri.getLabels(), targetLabels, threshold);
                if (scoreTypeObMax > objectSimilarity) {
                    keepObjectType = true;
                    objectSimilarity = scoreTypeObMax;
                }
            }

        } else if (tripleType != TripleType.OBJECT) {
            Set<String> hashObj = new HashSet<>();
            hashObj.add(object.toString());
            objectSimilarity = EmbeddingManager.similarity(hashObj, targetLabels, threshold);
        }


        return subjectSimilarity + predicateSimilarity + objectSimilarity;
    }

    public double compareLabel(INDArray targetLabels, double threshold, String targetEndpoint) {
        if (tripleType != TripleType.SUBJECT) {
            subjectType = subject.findMostSimilarType(targetEndpoint, targetLabels, threshold);
            double scoreTypeSubMax = 0;
            if (subjectType != null) {
                scoreTypeSubMax = EmbeddingManager.similarity(subjectType.getLabels(), targetLabels, threshold);
            }
            subjectSimilarity = EmbeddingManager.similarity(subject.getLabels(), targetLabels, threshold);
            if (scoreTypeSubMax > subjectSimilarity) {
                keepSubjectType = true;
                subjectSimilarity = scoreTypeSubMax;
            }
        }
        if (tripleType != TripleType.PREDICATE && !predicate.toString().equals("<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")) {
            predicateSimilarity = EmbeddingManager.similarity(predicate.getLabels(), targetLabels, threshold);
        }
        if (tripleType != TripleType.OBJECT && object instanceof IRI oiri) {
            objectType = oiri.findMostSimilarType(targetEndpoint, targetLabels, threshold);
            if (objectType != null) {
                double scoreTypeObMax = EmbeddingManager.similarity(objectType.getLabels(), targetLabels, threshold);
                objectSimilarity = EmbeddingManager.similarity(oiri.getLabels(), targetLabels, threshold);
                if (scoreTypeObMax > objectSimilarity) {
                    keepObjectType = true;
                    objectSimilarity = scoreTypeObMax;
                }
            }

        } else if (tripleType != TripleType.OBJECT) {
            Set<String> hashObj = new HashSet<>();
            hashObj.add(object.toString());
            objectSimilarity = EmbeddingManager.similarity(hashObj, targetLabels, threshold);
        }


        return subjectSimilarity + predicateSimilarity + objectSimilarity;
    }

    public double compareLabelEmb(INDArray targetLabels, double threshold, String targetEndpoint) {
        INDArray subjectEmb = null;
        INDArray predicateEmb = null;
        INDArray objectEmb = null;
        if (tripleType != TripleType.SUBJECT) {
            Set<String> subjectLabels = subject.getLabels();

            subjectType = subject.findMostSimilarType(targetEndpoint, targetLabels, threshold);
            if (subjectType != null) {
                subjectLabels.addAll(subjectType.getLabels());
            }
            subjectEmb = EmbeddingManager.embLabels(subjectLabels);
        }
        if (tripleType != TripleType.PREDICATE && !predicate.toString().equals("<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")) {
            predicateEmb = EmbeddingManager.embLabels(predicate.getLabels());
        }
        if (tripleType != TripleType.OBJECT && object instanceof IRI oiri) {
            Set<String> objectLabels = oiri.getLabels();

            objectType = oiri.findMostSimilarType(targetEndpoint, targetLabels, threshold);
            if (objectType != null) {
                objectLabels.addAll(objectType.getLabels());
            }
            objectEmb = EmbeddingManager.embLabels(objectLabels);
        } else if (tripleType != TripleType.OBJECT) {
            Set<String> hashObj = new HashSet<>();
            hashObj.add(object.toString());
            objectEmb = EmbeddingManager.embLabels(hashObj);
        }


        INDArray tripleEmb = Nd4j.zeros(EmbeddingManager.embshape);
        if (subjectEmb != null) {
            tripleEmb.addi(subjectEmb);
        }
        if (predicateEmb != null) {
            tripleEmb.addi(predicateEmb);
        }
        if (objectEmb != null) {
            tripleEmb.addi(objectEmb);
        }

        tripleEmb.divi(2);

        subjectSimilarity = EmbeddingManager.similarity(tripleEmb, targetLabels, threshold);
        return subjectSimilarity;
    }


    public IRI getSubject() {
        return subject;
    }

    public IRI getPredicate() {
        return predicate;
    }

    public Resource getObject() {
        return object;
    }

    public boolean isSubjectTriple() {
        return tripleType == TripleType.SUBJECT;
    }

    public boolean isPredicateTriple() {
        return tripleType == TripleType.PREDICATE;
    }

    public boolean isObjectTriple() {
        return tripleType == TripleType.OBJECT;
    }

    public String toString() {
        String subjStr = subject.toValueString();
        String predStr = predicate.toValueString();
        String objStr = object.toValueString();

        if (isSubjectTriple()) {
            subjStr = "?answer";
        } else if (isPredicateTriple()) {
            predStr = "?answer";
        } else if (isObjectTriple()) {
            objStr = "?answer";
        }

        return getResult(subjStr, predStr, objStr);
    }

    private String getResult(String subjStr, String predStr, String objStr) {
        String result = subjStr + " " + predStr + " " + objStr + ". ";
        if (keepSubjectType && !keepObjectType) {
            result = "?x " + predStr + " " + objStr + ". " +
                    "?x a " + subjectType + ". ";
        } else if (keepObjectType && !keepSubjectType) {
            result = subjStr + " " + predStr + " ?y. " +
                    "?y a " + objectType + ". ";
        } else if (keepObjectType) {
            result = "?x " + predStr + " ?y. " +
                    "?y a " + objectType + ". " +
                    "?x a " + subjectType + ". ";
        }
        return result;
    }


    public TripleType commonPartValue(Triple t) {
        TripleType res = TripleType.NONE;
        if (getType() == t.getType()) {
            if (getPredicate().equals(t.getPredicate()) && !isPredicateTriple()) {
                res = TripleType.PREDICATE;
            }
            if (getObject().equals(t.getObject()) && !isObjectTriple() && !keepObjectType) {
                res = TripleType.OBJECT;
            }
            if (getSubject().equals(t.getSubject()) && !isSubjectTriple() && !keepSubjectType) {
                res = TripleType.SUBJECT;
            }
        }
        return res;
    }

    public boolean hasCommonPart(Triple t) {
        boolean res = false;
        if (getType() == t.getType()) {
            if (!isSubjectTriple()) {
                res = getSubject().equals(t.getSubject());
            }
            if (!isPredicateTriple()) {
                res = res || getPredicate().equals(t.getPredicate());
            }
            if (!isObjectTriple()) {
                res = res || getObject().equals(t.getObject());
            }
        }
        return res;
    }


    public TripleType getType() {
        return tripleType;
    }

    public int hashCode() {
        return (subject.toString() + predicate.toString() + object.toString()).hashCode();
    }

    public boolean equals(Object obj) {
        if (obj instanceof Triple) {
            return (subject.toString() + predicate.toString() + object.toString())
                    .equals(((Triple) obj).subject.toString() + ((Triple) obj).predicate.toString() + ((Triple) obj).object.toString());
        } else {
            return false;
        }

    }

    public boolean isNullTriple() {
        return subject.toString().isEmpty() && predicate.toString().isEmpty() && object.toString().isEmpty();
    }


    public double getSimilarity() {
        return subjectSimilarity + predicateSimilarity + objectSimilarity;
    }


    public TripleType getPartGivingMaxSimilarity() {
        TripleType res = TripleType.NONE;
        if (subjectSimilarity > objectSimilarity && subjectSimilarity > predicateSimilarity) {
            res = TripleType.SUBJECT;
        } else if (objectSimilarity > subjectSimilarity && objectSimilarity > predicateSimilarity) {
            res = TripleType.OBJECT;
        } else if (predicateSimilarity > subjectSimilarity && predicateSimilarity > objectSimilarity) {
            res = TripleType.PREDICATE;
        }
        return res;

    }
}
