package org.deeplearning4j.examples.feedforward.classification;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.io.File;

/**
 * "Saturn" Data Classification Example
 *
 * Based on the data from Jason Baldridge:
 * 	https://github.com/jasonbaldridge/try-tf/tree/master/simdata
 *
 * @author Josh Patterson
 * @author Alex Black (added plots)
 *
 */
public class MLPClassifierSaturn {


    public static void main(String[] args) throws Exception {
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;
        int batchSize = 50;
        int seed = 123;
        double learningRate = 0.001;
        //Number of epochs (full passes of the data)
        int nEpochs = 30;

        int numInputs = 2;
        int numOutputs = 2;
        int numHiddenNodes = 20;

        final String filenameTrain  = new ClassPathResource("/classification/saturn_data_train.csv").getFile().getPath();
        final String filenameTest  = new ClassPathResource("/classification/saturn_data_eval.csv").getFile().getPath();

        //Load the training data:
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File(filenameTrain)));

        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr,batchSize,0,2);

        //Load the test/evaluation data:
        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File(filenameTest)));

        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest,batchSize,0,2);

        //log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .updater(new Nesterovs(learningRate, 0.9))
            .list()
            .layer(0, new DenseLayer.Builder()
                .nIn(numInputs)
                .nOut(numHiddenNodes)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.RELU)
                .build())
            .layer(1, new OutputLayer.Builder()
                .nIn(numHiddenNodes)
                .nOut(numOutputs)
                .lossFunction(LossFunction.NEGATIVELOGLIKELIHOOD)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.SOFTMAX)
                .build())
            .pretrain(false)
            .backprop(true)
            .build();


        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));    //Print score every 10 parameter updates

        for ( int n = 0; n < nEpochs; n++) {
            model.fit( trainIter );
        }

        System.out.println("Evaluate model....");
        Evaluation eval = new Evaluation(numOutputs);
        while(testIter.hasNext()){
            DataSet t = testIter.next();
            INDArray features = t.getFeatureMatrix();
            INDArray lables = t.getLabels();
            INDArray predicted = model.output(features,false);

            eval.eval(lables, predicted);

        }

        System.out.println("TP: " + eval.getTruePositives().totalCount());
        System.out.println("TN: " + eval.getTrueNegatives().totalCount());
        System.out.println("FP: " + eval.getFalsePositives().totalCount());
        System.out.println("FN: " + eval.getFalseNegatives().totalCount());

        System.out.println(eval.stats());

    }

}
