package org.deeplearning4j.examples.feedforward.classification.wineClassification.old.getWineQuality;

import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class PredictWineQuality {

    static int seed = 123;
    static int batchSize = 100;
    static int numInputs = 11;
    static int numClasses = 10;//7; //[3, 9]
    static int epoch = 40;
    static double learningRate = 0.01;
    static final double trainingRatio = 0.6; //the amount of data per file


    public static void main(String[] args) throws Exception {
        char delimiter = ';';
        int numLinesToSkip = 1;

        WineQualityRecordReader rrTrain = new WineQualityRecordReader(numLinesToSkip, delimiter, trainingRatio, true);
        WineQualityRecordReader rrTest = new WineQualityRecordReader(numLinesToSkip, delimiter, 1 - trainingRatio, false);

        String rootDir = new ClassPathResource("classification/wineQuality").getFile().getAbsolutePath();
        FileSplit fileSplit = new FileSplit(new File(rootDir), new String[]{"csv"});

        rrTrain.initialize(fileSplit);
        rrTest.initialize(fileSplit);

        DataSetIterator trainIter = new RecordReaderDataSetIterator(rrTrain, batchSize, 11, numClasses);
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest, batchSize, 11, numClasses);

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(learningRate, Nesterovs.DEFAULT_NESTEROV_MOMENTUM))
                .l2(0.001)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(numInputs)
                        .nOut(250)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(250)
                        .nOut(200)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new DenseLayer.Builder()
                        .nIn(200)
                        .nOut(150)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new OutputLayer.Builder()
                        .nIn(150)
                        .nOut(numClasses)
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();

        MultiLayerNetwork network = new MultiLayerNetwork(config);


        network.setListeners(new ScoreIterationListener(10));

        for(int i = 0; i < epoch; ++i)
        {
            network.fit(trainIter);
        }

        Evaluation eval = network.evaluate(testIter);

        System.out.println(eval.stats());


    }
}
