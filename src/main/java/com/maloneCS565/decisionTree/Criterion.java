package decisionTree;

import decisionTree.*;

import java.util.*;
import weka.core.*;

public abstract class Criterion {

    public final Attribute attr;

    public Criterion(Attribute attr) {
       this.attr = attr; 
    }

    public abstract boolean matches(Instance inst);

    public abstract String toString();
}
