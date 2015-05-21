package org.numenta.nupic.algorithms;

import org.numenta.nupic.util.Tuple;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Date: 5/7/15
 */
public class KnnClassifier {

    int verbosity = 0;

    /**
     * History of the last _maxSteps activation patterns. We need to keep
     * these so that we can associate the current iteration's classification
     * with the activationPattern from N steps ago
     */
    org.numenta.nupic.util.Deque<Tuple> patternNZHistory;

    /**
     * The bit's learning iteration. This is updated each time store() gets
     * called on this bit.
     */
    int learnIteration;
    /**
     * This contains the offset between the recordNum (provided by caller) and
     * learnIteration (internal only, always starts at 0).
     */
    int recordNumMinusLearnIteration = -1;


    String g_debugPrefix = "KNNClassifier";

    /**
     * Used to record the association between an observed input and a category
     */
    public class Observation {
        public final int[] input;
        public final Object category;

        public Observation(int[] input, Object category) {
            this.input = input;
            this.category = category;
        }
    }

    /**
     * An observation that has a known distance from an unclassified input
     */
    public class ObservationWithDistance extends Observation {
        public final double distance;

        public ObservationWithDistance(Observation o, double distance) {
            super(o.input, o.category);
            this.distance = distance;
        }
    }

    /**
     * Todo: implement manattan distance as option
     */
    public enum DistanceCalc {
        Euclidian,
        Manhattan
    }

    /**
     * The number of the closest distances to consider when counting.  T
     */
    private final int k;

    /**
     * The learned memory, all observed patterns.
     * Todo:  this may need to be the history, a queue that only keeps the past N observations
     * todo:  otherwise this classification will become not useful at scale
     */
    private final List<Observation> patterns = new ArrayList<Observation>();


    /**
     * Creates a classifier
     * @param k
     */
    public KnnClassifier(int k) {
        this.k = k;
    }


    /**
     * Process one input sample.
     * This method is called by outer loop code outside the nupic-engine. We
     * use this instead of the nupic engine compute() because our inputs and
     * outputs aren't fixed size vectors of reals.
     *
     * @param recordNum      Record number of this input pattern. Record numbers should
     *                       normally increase sequentially by 1 each time unless there
     *                       are missing records in the dataset. Knowing this information
     *                       insures that we don't get confused by missing records.
     * @param classification {@link Map} of the classification information:
     *                       bucketIdx: index of the encoder bucket
     *                       actValue:  actual value going into the encoder
     * @param patternNZ      list of the active indices from the output below
     * @param learn          if true, learn this sample
     * @param infer          if true, perform inference
     * @return                 {@link KnnClassifierResult} of the classification
     */
    public KnnClassifierResult compute(int recordNum, Map<String, Object> classification, int[] patternNZ, boolean learn, boolean infer) {

        if (recordNumMinusLearnIteration == -1) {
            recordNumMinusLearnIteration = recordNum - learnIteration;
        }

        // Update the learn iteration
        learnIteration = recordNum - recordNumMinusLearnIteration;

        if (verbosity >= 1) {
            System.out.println(String.format("\n%s: compute ", g_debugPrefix));
            System.out.println(" recordNum: " + recordNum);
            System.out.println(" learnIteration: " + learnIteration);
            System.out.println(String.format(" patternNZ(%d): ", patternNZ.length, patternNZ));
            System.out.println(" classificationIn: " + classification);
        }

        patternNZHistory.append(new Tuple(learnIteration, patternNZ));

        KnnClassifierResult result = null;

        if (infer) {

            result = infer(patternNZ);

        }

        if (learn && classification.get("bucketIdx") != null) {
            // Get classification info
            Object actValue = classification.get("actValue");

            learn(patternNZ, actValue);
        }

        return result;


    }

    public KnnClassifierResult infer(int[] columns) {

        if (patterns.isEmpty()) {
            return null;
        }

        List<ObservationWithDistance> observedDistances = patterns.stream().map(x -> new ObservationWithDistance(x, getDistance(x.input, columns)))
                .sorted((o1, o2) -> Double.compare(o1.distance, o2.distance))
                .collect(Collectors.toList());


        Map<Object, Integer> inferenceResult = new HashMap<>();

        Map<Object, Double> closestToEachCategory = new HashMap<>();
        double[] distances = new double[observedDistances.size()];

        for (int i = 0; i < observedDistances.size(); i++) {

            ObservationWithDistance o = observedDistances.get(i);

            distances[i] = o.distance;

            if (i < k) {
                //counted in scoring for winner
                if (!inferenceResult.containsKey(o.category)) {
                    inferenceResult.put(o.category, 1);
                } else {
                    int previous = inferenceResult.get(o.category);
                    inferenceResult.put(o.category, previous + 1);
                }

            }

            double previousLow = closestToEachCategory.getOrDefault(o.category, Double.MAX_VALUE);

            if (o.distance < previousLow) {
                closestToEachCategory.put(o.category, o.distance);
            }
        }

        //winner is whichever categoryWinnerVotes is the highest
        Object winner = inferenceResult.entrySet().stream()
                .sorted((o1, o2) -> o2.getValue().compareTo(o1.getValue()))
                .findFirst().get().getKey();

        return new KnnClassifierResult(winner, inferenceResult, closestToEachCategory, distances);
    }


    double getDistance(int[] test, int[] candidate) {

        assert test.length == candidate.length;

        Double accumulator = 0d;

        for (int i = 0; i < test.length; i++) {

            accumulator += Math.pow(test[i] - candidate[i], 2);


        }

        return Math.sqrt(accumulator);
    }

    public void learn(int[] key, Object value) {

        this.patterns.add(new Observation(key, value));

    }

}
