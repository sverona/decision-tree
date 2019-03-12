package decisionTree;

import decisionTree.*;

import java.util.*;
import weka.core.*;

public class EntropyDecisionTree extends DecisionTree {

    public final double minEntropy;

    public EntropyDecisionTree(Instances data) {
        this(data, 0.3);
    }

    public EntropyDecisionTree(Instances data, double minEntropy) {
        super(data);
        this.minEntropy = minEntropy;
    }

    private double gain(DecisionTreeNode node, int splitIdx) {
        Instances data = node.getData();
        
        List<Double> classes = node.attrValues(data.classAttribute());

        List<Double> values = new ArrayList<Double>();
        int numValues = 1;
        if (splitIdx >= 0) {
            values = node.attrValues(splitIdx);
            numValues = values.size();
        }
         
        int[][] histogram = new int[classes.size()][numValues];
        for (Instance d : data) {
            int valueIdx = splitIdx >= 0 ? values.indexOf(d.value(splitIdx)) : 0;
            int classIdx = classes.indexOf(d.classValue());

            histogram[classIdx][valueIdx]++;
        }  

        double entropy = 0d;
        for (int classIdx = 0; classIdx < classes.size(); classIdx++) {
            double p = 0;
            for (int valueIdx = 0; valueIdx < numValues; valueIdx++) {
                p += histogram[classIdx][valueIdx];
            }
            p /= data.size();
            if (p > 0) {
                entropy -= p * Math.log10(p);
            }
        }

        if (splitIdx < 0) {
            return entropy;
        } else {
            double[] childrenEntropy = new double[numValues];
            for (int valueIdx = 0; valueIdx < numValues; valueIdx++) {
                double total = 0d;
                for (int classIdx = 0; classIdx < classes.size(); classIdx++) {
                    total += histogram[classIdx][valueIdx];
                }

                for (int classIdx = 0; classIdx < classes.size(); classIdx++) {
                    double p = histogram[classIdx][valueIdx] / total;
                    if (p > 0) {
                        childrenEntropy[valueIdx] -= p * Math.log10(p);
                    }
                }
            }

            double mean = 0d;
            for (double d : childrenEntropy) {
                mean += d;
            }
            mean /= numValues;

            return entropy - mean;
        }
    }

    protected void printChildren(DecisionTreeNode node, int depth) {
        for (int i = 0; i < depth; i++) {
            System.out.print("  ");
        }
        System.out.println(node.edgeLabel);
        if (node.getChildren().isEmpty()) {
            for (int i = 0; i <= depth; i++) {
                System.out.print("  ");
            }
            System.out.printf("%s (%d, %f)\n", node.getData().classAttribute().value((int)node.classLabel), node.getData().size(), this.gain(node, -1));
        }
        for (DecisionTreeNode child : node.getChildren()) {
            printChildren(child, depth + 1);
        }
    }

    protected boolean stoppingCond(DecisionTreeNode node) {
        double entropy = this.gain(node, -1);
        return entropy < this.minEntropy;
    }

    protected List<Criterion> findBestSplit(DecisionTreeNode node) {
        Instances data = node.getData();

        List<Criterion> splits = new ArrayList<Criterion>();
        double maxGain = -1d;
        int maxAttrIdx = data.classIndex() == 0 ? 1 : 0;
        for (int attrIdx = 0; attrIdx < data.numAttributes(); attrIdx++) {
            if (attrIdx != data.classIndex()) {
                double thisGain = this.gain(node, attrIdx);
                if (thisGain > maxGain) {
                    maxGain = thisGain;
                    maxAttrIdx = attrIdx;
                }
            }
        }

        Attribute bestAttr = data.attribute(maxAttrIdx);
        for (Double val : node.attrValues(maxAttrIdx)) {
            splits.add(new PointCriterion(bestAttr, val.doubleValue()));
        }

        return splits;
    }
}
