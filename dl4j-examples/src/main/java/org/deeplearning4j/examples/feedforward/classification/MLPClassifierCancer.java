package org.deeplearning4j.examples.feedforward.classification;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.api.storage.StatsStorage;
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
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class MLPClassifierCancer
{
    static int labelIndex = 9;
    static int numClasses = 2;
    static int batchSize = 38;

    static int seedNumber = 123;
    static int trainDataSize = 380;
    static int testDataSize = 80;

    static int numInputs = labelIndex;

    private static Logger log = LoggerFactory.getLogger(MLPClassifierCancer.class);

    public static void main(String[] args) throws Exception
    {
        File trainDataFile = new ClassPathResource("/classification/BreastCancerData_train.csv").getFile();
        File testDataFile = new ClassPathResource("/classification/BreastCancerData_test.csv").getFile();

        /*
        RecordReader rrTrain = new CSVRecordReader(1, ",");
        rrTrain.initialize(new FileSplit(trainDataFile));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rrTrain, batchSize, labelIndex, numClasses);

        RecordReader rrTest = new CSVRecordReader(1, ",");
        rrTest.initialize(new FileSplit(testDataFile));
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest, batchSize, labelIndex, numClasses);
        */

        //Get Data Iterator
        DataSetIterator trainIter = getShuffledData(trainDataFile, trainDataSize);
        DataSetIterator testIter = getShuffledData(testDataFile, testDataSize);

        //Network Config
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seedNumber)
                .updater(new Nesterovs(0.001, Nesterovs.DEFAULT_NESTEROV_MOMENTUM))
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(numInputs)
                        .nOut(100)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new OutputLayer.Builder()
                        .nIn(100)
                        .nOut(numClasses)
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX)
                        .build())
                .pretrain(false)
                .backprop(true)
                .build();


        //UI Score Interface set up

        UIServer server = UIServer.getInstance();
        StatsStorage storage = new InMemoryStatsStorage();
        server.attach(storage);


        //Network initialization
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        model.setListeners(new ScoreIterationListener(10), new StatsListener(storage, 10));

        //Start training

        int epoch = 100;

        int evaluateCycle = 5;
        for(int i = 0; i < epoch; ++i)
        {
            model.fit(trainIter);

            if(i % evaluateCycle == 0)
            {
                Evaluation eval = new Evaluation(numClasses);

                while(testIter.hasNext())
                {
                    DataSet dt = testIter.next();

                    INDArray feature = dt.getFeatureMatrix();
                    INDArray predicted = model.output(feature);

                    eval.eval(dt.getLabels(), predicted);
                }

                System.out.println("Epoch: " + i);
                System.out.println(eval.accuracy());
            }
            testIter.reset();

        }




        System.out.println("Final Evaluation");

        Evaluation eval = new Evaluation(numClasses);

        while(testIter.hasNext())
        {
            DataSet dt = testIter.next();

            INDArray feature = dt.getFeatureMatrix();
            INDArray predicted = model.output(feature);

            eval.eval(dt.getLabels(), predicted);
        }

        System.out.println(eval.stats());

        testIter.reset();
    }


    /**
     * Specially shuffle the data before passing into DataSetIterator
     * Looking into datafile will realise data labels are organized.
     * Shuffle to make each batch size have labels of different class
     *
     * @param dataFile: file name
     * @param totalSampleSize: hardcoded total data file by looking at total data in a file
     * @return
     * @throws Exception
     */
    private static DataSetIterator getShuffledData(File dataFile, int totalSampleSize) throws Exception
    {
        int numLinesToSkip = 1;
        String delimiter = ",";

        //Read from CSV file
        RecordReader rr = new CSVRecordReader(numLinesToSkip, delimiter);
        rr.initialize(new FileSplit(dataFile));

        //Get all data before shuffle
        DataSetIterator iter = new RecordReaderDataSetIterator(rr, totalSampleSize, labelIndex, numClasses);

        List<DataSet> dataList = iter.next().asList();

        //Shuffle data
        Collections.shuffle(dataList, new Random(seedNumber));

        return new ListDataSetIterator(dataList, batchSize);
    }
}
