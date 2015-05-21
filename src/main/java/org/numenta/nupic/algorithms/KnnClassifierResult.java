package org.numenta.nupic.algorithms;

import java.util.Map;

/**
 * Date: 5/21/15
 */
public class KnnClassifierResult {

    /**
     * The category chosen as the closest to the input
     */
    private final Object winner;

    /**
     * list with length numCategories
     * Each entry contains the number of neighbors within the
     * top K neighbors that are in that category
     */
    private final Map<Object, Integer> inferenceResult;

    /**
     *  Distance from each learned example to the passed in sample
     *  Normalized from 0 to 1
     */
    private final double[] distances;

    /**
     * Distance from the unknown to the nearest example of each category
     */
    private final Map<Object, Double> categoryDistance;

    public KnnClassifierResult(Object winner, Map<Object, Integer> inferenceResult, Map<Object, Double> closestToEachCategory, double[] distances) {
        this.winner = winner;
        this.inferenceResult = inferenceResult;
        this.categoryDistance = closestToEachCategory;
        this.distances = distances;

    }


    public Object getWinner(){
        return winner;
    }


    public Map<Object, Integer> getInferenceResult() {
        return inferenceResult;
    }

    public double getClosestToCategory(Object category) {

        return categoryDistance.getOrDefault(category, null);
    }
}