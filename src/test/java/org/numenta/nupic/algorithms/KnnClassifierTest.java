package org.numenta.nupic.algorithms;

import org.junit.Assert;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

/**
 * Date: 5/8/15
 */
public class KnnClassifierTest {

    @Test
    public void testGetDistance(){

        KnnClassifier k = new KnnClassifier(1);

        int[] a = new int[] {0, 1, 1, 1, 0};
        int[] b = new int[] {0, 1, 1, 1, 0};

        Assert.assertEquals(0, k.getDistance(a, b), 0.001);


        int[] c = new int[] {0, 1, 1, 1, 0};
        int[] d = new int[] {0, 1, 0, 0, 1};


        Assert.assertEquals(Math.sqrt(3.0), k.getDistance(c, d), 0.001);

    }


    @Test
    public void testKnnClassifier(){

        KnnClassifier classifier = new KnnClassifier(3);


        Double[] test = new Double[] {1.0, 1.4, 0.9, 1.1, 1.0};

        classifier.learn(new int[] {0,0,0,0,1,1,1,1}, "one");
        classifier.learn(new int[] {0,0,0,0,1,1,0,1}, "one");
        classifier.learn(new int[] {0,0,0,1,1,1,1,0}, "one");
        classifier.learn(new int[] {1,1,1,1,0,0,0,0}, "two");
        classifier.learn(new int[] {1,1,1,1,1,0,0,0}, "two");
        classifier.learn(new int[] {0,1,1,1,0,0,0,0}, "two");
        classifier.learn(new int[] {1,0,1,1,0,0,0,0}, "two");

        KnnClassifierResult result = classifier.infer(new int[]{0,0,0,0,1,1,1,1});

        Assert.assertEquals("one", result.getWinner());
        Assert.assertEquals(Math.sqrt(7), result.getClosestToCategory("two"), 0.0001);
        Assert.assertEquals(3, result.getInferenceResult().get("one").intValue());


    }



}
