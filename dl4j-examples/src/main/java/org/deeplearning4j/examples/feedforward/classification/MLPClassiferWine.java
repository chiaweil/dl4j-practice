package org.deeplearning4j.examples.feedforward.classification;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
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
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * 0 is white, 1 is red
 */
public class MLPClassiferWine
{
    static int seed = 123;
    static int batchSize = 100;
    static int numInputs = 12;
    static int numClasses = 2;
    static int epoch = 40;
    static double learningRate = 0.01;
    static double splitRatio = 0.8;
    static final int MINEXAMPLESPERCLASS = 1599; //the amount of data per file

    public static void main(String[] args) throws Exception
    {
        List<String> wineLabels = Arrays.asList(new String[]{"White Wine", "Red Wine"});
        File dataFile = new ClassPathResource("/classification/wineData.csv").getFile();

        int numLinesToSkip = 1;
        char delimiter = ',';

        //Read from CSV file
        RecordReader rr = new CSVRecordReader(numLinesToSkip, delimiter);
        rr.initialize(new FileSplit(dataFile));

        //Get Data Iterator
        //Get all data before shuffle
        DataSetIterator bufferIter = new RecordReaderDataSetIterator(rr, MINEXAMPLESPERCLASS * 2, numInputs, numClasses);

        List<DataSet> bufferList = bufferIter.next().asList();

        //Shuffle data
        Collections.shuffle(bufferList, new Random(seed));

        //Split training and testing dataIter

        int trainEndIndex = (int) (Math.ceil(bufferList.size() * splitRatio));
        List<DataSet> trainingList = bufferList.subList(0, trainEndIndex);
        List<DataSet> testingList = bufferList.subList(trainEndIndex, bufferList.size());

        DataSetIterator trainIter = new ListDataSetIterator(trainingList, batchSize);
        DataSetIterator testIter = new ListDataSetIterator(testingList, batchSize);


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
        eval.setLabelsList(wineLabels);

        System.out.println(eval.stats());



    }
}
