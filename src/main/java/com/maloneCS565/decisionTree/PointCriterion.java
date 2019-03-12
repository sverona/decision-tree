package decisionTree;

import decisionTree.*;

import java.util.*;
import weka.core.*;

public class PointCriterion extends Criterion {

    public final double value;

    public PointCriterion(Attribute attr, double value) {
        super(attr);
        this.value = value;
    }

    public boolean matches(Instance inst) {
        return inst.value(this.attr) == this.value;
    }

    public String toString() {
        return this.attr.name() + " = " + attr.value((int)this.value);
    }
}
