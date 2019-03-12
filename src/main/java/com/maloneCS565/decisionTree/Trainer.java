package decisionTree;

import decisionTree.*;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import weka.core.*;
import weka.core.converters.ArffLoader.ArffReader;

public class Trainer {
    public static void main(String[] args) throws IOException {

        DecisionTree tree;

        if (args.length == 0) {
            System.out.println("usage: java Trainer.jar [data.arff]");
            System.exit(1);
        }

        BufferedReader buf = new BufferedReader(new FileReader(args[0]));
        ArffReader in = new ArffReader(buf);
        Instances data = in.getData();
        data.setClass(data.attribute("class"));

        if (data.classIndex() < 0) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        // 2-fold cross-validation
        for (int i = 0; i < 2; i++) {
           Instances trainingSet = data.trainCV(2, i); 
           Instances testSet = data.testCV(2, i);

           // tree = new RandomizedDecisionTree(trainingSet);
           tree = new EntropyDecisionTree(trainingSet, 0.2);
           tree.train();

           tree.printTree();

           double accuracy = 0;
           for (Instance inst : testSet) {
               if (tree.classify(inst) == inst.classValue()) {
                   accuracy++;
               }
           }
           accuracy /= testSet.size();
           System.out.println(accuracy);
        }

    }
}
