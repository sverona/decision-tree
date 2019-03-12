package decisionTree;

import decisionTree.*;

import java.util.*;
import weka.core.*;

public class IntervalCriterion extends Criterion {

    public final double minValue;
    public final double maxValue;

    public final boolean includesMin;
    public final boolean includesMax;

    public IntervalCriterion(Attribute attr, double minValue, double maxValue, boolean includesMin, boolean includesMax) {
        super(attr);
        this.minValue = minValue;
        this.maxValue = maxValue;
        this.includesMin = includesMin;
        this.includesMax = includesMax;
    }

    public boolean matches(Instance inst) {
        double value = inst.value(this.attr);
        if (value == this.minValue) {
            return this.includesMin;
        } else if (value == this.maxValue) {
            return this.includesMax;
        } else {
            return this.minValue < value && value < this.maxValue;
        }
    }

    public String toString() {
        return Double.valueOf(minValue).toString() +
               (includesMin ? " <= " : " < ") +
               this.attr.name() +
               (includesMax ? " <= " : " < ") +
               Double.valueOf(maxValue).toString();
    }
}
