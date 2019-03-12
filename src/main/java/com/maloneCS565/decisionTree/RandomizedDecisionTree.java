package decisionTree;

import decisionTree.*;

import java.util.*;
import weka.core.*;

public class RandomizedDecisionTree extends DecisionTree {

    Random rng = new Random();

    public RandomizedDecisionTree(Instances data) {
        super(data);
    }

    protected boolean stoppingCond(DecisionTreeNode node) {
        Instances data = node.getData();
        Map.Entry<Double, Integer> mode = this.mode(node);

        return mode.getValue() * 2 >= data.size();
    }

    protected List<Criterion> findBestSplit(DecisionTreeNode node) {
        Instances data = node.getData();
        int splitIdx;
        while (true) {
            splitIdx = rng.nextInt(data.numAttributes());
            if (splitIdx != data.classIndex()) {
                AttributeStats stats = data.attributeStats(splitIdx);
                if (stats.distinctCount > 1) {
                    break;
                }
            }
        }
        List<Criterion> splits = new ArrayList<Criterion>();

        Attribute bestAttr = data.attribute(splitIdx);
        for (Double val : node.attrValues(splitIdx)) {
            splits.add(new PointCriterion(bestAttr, val.doubleValue()));
        }

        return splits;
    }
}
