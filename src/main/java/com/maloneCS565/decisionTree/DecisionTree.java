package decisionTree;

import decisionTree.*;

import java.util.*;
import weka.core.*;

public abstract class DecisionTree {

    public final DecisionTreeNode root;

    public DecisionTree(Instances data) {
        this.root = new DecisionTreeNode(); 
        this.root.setData(data);

        this.root.edgeLabel = "root";
    }

    public void train() {
        /* Clear/re-initialize tree */
        growTree(this.root);
    }

    protected Map.Entry<Double, Integer> mode(DecisionTreeNode node) {
	    /* Determine which class label is the most frequent at this node.
         * Return the label. If there is more than one mode, return the class label that is ordered first.
         */
        Instances data = node.getData();
        HashMap<Double, Integer> freqs = node.histogram(data.classAttribute());

        Map.Entry<Double, Integer> maxPair = Collections.max(freqs.entrySet(),
              (k1, k2) -> (k1.getValue() - k2.getValue()));

        return maxPair;
    }

    protected boolean isHomogeneous(DecisionTreeNode node) {
        Instances data = node.getData();
        InstanceComparator c = new InstanceComparator(false);

        Instance first = data.get(0);
        for (int idx = 1; idx < data.size(); idx++) {
            Instance other = data.get(idx);
            if (c.compare(first, other) != 0) {
                return false;
            }
        }
        return true;
    }

    protected double chooseLabel(DecisionTreeNode node) {
        if (node.getData().size() == 0) {
            return 0;
        }
        int valIndex = (int)this.mode(node).getKey().doubleValue();
        return valIndex;
    }

    protected abstract boolean stoppingCond(DecisionTreeNode node);

    protected abstract List<Criterion> findBestSplit(DecisionTreeNode node);

    public void printTree() {
        printChildren(this.root, 0);
    }

    protected void printChildren(DecisionTreeNode node, int depth) {
        /* Preorder traversal */
        for (int i = 0; i < depth; i++) {
            System.out.print("  ");
        }
        System.out.println(node.edgeLabel);
        if (node.getChildren().isEmpty()) {
            for (int i = 0; i <= depth; i++) {
                System.out.print("  ");
            }
            System.out.printf("%s (%d)\n", node.getData().classAttribute().value((int)node.classLabel), node.getData().size());
        }
        for (DecisionTreeNode child : node.getChildren()) {
            printChildren(child, depth + 1);
        }
    }

    protected void growTree(DecisionTreeNode node) {
        if (this.stoppingCond(node) || this.isHomogeneous(node)) {
            /* We can stop */
            // System.out.println(chooseLabel(node));
            Instances data = node.getData();
            node.classLabel = chooseLabel(node);
            node.criterion = new PointCriterion(data.classAttribute(), node.classLabel);
        } else {
            /* Find a split */    
            List<Criterion> split = findBestSplit(node);

            List<DecisionTreeNode> children = new ArrayList<DecisionTreeNode>();
            for (Criterion c : split) {
                DecisionTreeNode thisChild = new DecisionTreeNode();
                children.add(thisChild);
            }

            Instances data = node.getData();
            for (int idx = 0; idx < split.size(); idx++) {
                Criterion c = split.get(idx);
                Instances childData = new Instances(data, 0, 0);

                for (Instance inst : data) {
                    if (c.matches(inst)) {
                        childData.add(inst);
                    }
                }

                DecisionTreeNode thisChild = children.get(idx);
                thisChild.criterion = c;
                thisChild.setEdgeLabel(c.toString());
                thisChild.setData(childData);
            }

            node.setChildren(children);

            for (DecisionTreeNode child : children) {
                growTree(child);
            }
        }
    }

    protected double classify(Instance inst) {
        DecisionTreeNode curNode = this.root;
        while (!curNode.getChildren().isEmpty()) {
            boolean childFound = false;
            for (DecisionTreeNode child : curNode.getChildren()) {
                if (!childFound && child.criterion.matches(inst)) {
                    curNode = child;
                    childFound = true;
                }
            }
        }
        return curNode.classLabel;
    }
}
