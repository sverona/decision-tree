package decisionTree;

import decisionTree.*;

import java.util.*;
import weka.core.*;

public class DecisionTreeNode {

    protected double classLabel;
    protected String edgeLabel;
    protected LinkedList<DecisionTreeNode> children = new LinkedList<DecisionTreeNode>();
    protected Instances data;
    public Criterion criterion;

    /* constructors */

    public DecisionTreeNode() {}

    public DecisionTreeNode(Instances data) {
        this.data = data;
    }

    /* getters and setters */

    public double getClassLabel() {
        return this.classLabel;
    }

    public void setClassLabel(double classLabel) {
        this.classLabel = classLabel;
    }

    public String getEdgeLabel() {
        return this.edgeLabel;
    }

    public void setEdgeLabel(String edgeLabel) {
        this.edgeLabel = edgeLabel;
    }

    public List<DecisionTreeNode> getChildren() {
        return this.children;
    }

    public void setChildren(List<DecisionTreeNode> children) {
        this.children = new LinkedList<DecisionTreeNode>();

        this.children.addAll(children);
    }

    public Instances getData() {
        return this.data;
    }

    public void setData(Instances data) {
        this.data = new Instances(data);
    }

    /* end getters and setters */

    public HashMap<Double, Integer> histogram(Attribute attr) {
        HashMap<Double, Integer> hist = new HashMap<Double, Integer>();

        for (Instance d : this.data) {
            Double attrVal = d.value(attr);
            int freq = hist.getOrDefault(attrVal, 0);
            hist.put(attrVal, freq + 1);
        }

        /* for (Map.Entry<Double, Integer> e : hist.entrySet()) {
            System.out.printf("%s %d\n", e.getKey(), e.getValue());
        }
        */

        return hist;
    }

    public List<Double> attrValues(int attrIdx) {
        return this.attrValues(this.data.attribute(attrIdx));
    }

    public List<Double> attrValues(Attribute attr) {
        List<Double> values = new ArrayList<Double>();
        values.addAll(this.histogram(attr).keySet());

        return values;
    }
}
