package org.deeplearning4j.examples.recurrent.seqclassification;

import com.google.flatbuffers.FlatBufferBuilder;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.INDArrayDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.layers.normalization.BatchNormalizationHelper;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ndarray.SparseFormat;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.exception.Nd4jNoSuchWorkspaceException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.ShapeOffsetResolution;
import org.nd4j.linalg.indexing.conditions.Condition;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.net.URL;
import java.nio.IntBuffer;
import java.util.*;

/**
 * Sequence Classification Example Using a LSTM Recurrent Neural Network
 *
 * This example learns how to classify univariate time series as belonging to one of six categories.
 * Categories are: Normal, Cyclic, Increasing trend, Decreasing trend, Upward shift, Downward shift
 *
 * Data is the UCI Synthetic Control Chart Time Series Data Set
 * Details:     https://archive.ics.uci.edu/ml/datasets/Synthetic+Control+Chart+Time+Series
 * Data:        https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/synthetic_control.data
 * Image:       https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/data.jpeg
 *
 * This example proceeds as follows:
 * 1. Download and prepare the data (in downloadUCIData() method)
 *    (a) Split the 600 sequences into train set of size 450, and test set of size 150
 *    (b) Write the data into a format suitable for loading using the CSVSequenceRecordReader for sequence classification
 *        This format: one time series per file, and a separate file for the labels.
 *        For example, train/features/0.csv is the features using with the labels file train/labels/0.csv
 *        Because the data is a univariate time series, we only have one column in the CSV files. Normally, each column
 *        would contain multiple values - one time step per row.
 *        Furthermore, because we have only one label for each time series, the labels CSV files contain only a single value
 *
 * 2. Load the training data using CSVSequenceRecordReader (to load/parse the CSV files) and SequenceRecordReaderDataSetIterator
 *    (to convert it to DataSet objects, ready to train)
 *    For more details on this step, see: http://deeplearning4j.org/usingrnns#data
 *
 * 3. Normalize the data. The raw data contain values that are too large for effective training, and need to be normalized.
 *    Normalization is conducted using NormalizerStandardize, based on statistics (mean, st.dev) collected on the training
 *    data only. Note that both the training data and test data are normalized in the same way.
 *
 * 4. Configure the network
 *    The data set here is very small, so we can't afford to use a large network with many parameters.
 *    We are using one small LSTM layer and one RNN output layer
 *
 * 5. Train the network for 40 epochs
 *    At each epoch, evaluate and print the accuracy and f1 on the test set
 *
 * @author Alex Black
 */

public class UCISequenceClassification
{
    private static Logger log = LoggerFactory.getLogger(UCISequenceClassification.class);

    //'baseDir': Base directory for the data. Change this if you want to save the data somewhere else
    private static File baseDir = new File("dl4j-examples/src/main/resources/uci/");
    private static File baseTrainDir = new File(baseDir, "train");
    private static File featuresDirTrain = new File(baseTrainDir, "features");
    private static File labelsDirTrain = new File(baseTrainDir, "labels");
    private static File baseTestDir = new File(baseDir, "test");
    private static File featuresDirTest = new File(baseTestDir, "features");
    private static File labelsDirTest = new File(baseTestDir, "labels");

    private static final int trainSamples = 450;//75% train, 25% test
    private static final int testSamples = 150;

    private static final int BATCH_SIZE = 50;
    private static final int NUM_CLASSES = 6;

    public static void main(String[] args) throws Exception
    {

        double m[][][] = {{
            {1,2,3,4},
            {5,6,7,8},
            {9,10,11,12},
        }};

        INDArray arr1 = Nd4j.create(m);

        INDArray arr2 = arr1.getColumn(0);//arr1.get(NDArrayIndex.all(), NDArrayIndex.interval(1,2)).getRow(0);

        double arr3 = arr2.getDouble(1,2);

        System.out.println("output: arr3");

        //change baseDir
        try
        {
            String[] filePath = baseDir.toString().split("/");
            baseDir = new ClassPathResource(filePath[filePath.length - 1]).getFile();

        }
        catch(Exception e)
        {
            //download Data
            downloadUCIData();
        }

        SequenceRecordReader rrTrainFeature = new CSVSequenceRecordReader();
        rrTrainFeature.initialize(new NumberedFileInputSplit(featuresDirTrain.getAbsolutePath().toString() + "/%d.csv", 0, trainSamples - 1));

        SequenceRecordReader rrTrainLabel = new CSVSequenceRecordReader();
        rrTrainLabel.initialize(new NumberedFileInputSplit(labelsDirTrain.getAbsolutePath().toString() + "/%d.csv", 0, trainSamples - 1));

        DataSetIterator trainIter = new SequenceRecordReaderDataSetIterator(rrTrainFeature, rrTrainLabel, BATCH_SIZE, NUM_CLASSES, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);


        //Apply the normalization to
        //DataSetIterator trainIter and testIter
        //Normalize the training data
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainIter); //Collect training data statistics
        trainIter.reset();

        //Use previously collected statistics to normalize on-the-fly. Each DataSet returned by 'trainData' iterator will be normalized
        trainIter.setPreProcessor(normalizer);


        SequenceRecordReader rrTestFeature = new CSVSequenceRecordReader();
        rrTestFeature.initialize(new NumberedFileInputSplit(featuresDirTest.getAbsolutePath().toString() + "/%d.csv", 0, testSamples - 1));

        SequenceRecordReader rrTestLabel = new CSVSequenceRecordReader();
        rrTestLabel.initialize(new NumberedFileInputSplit(labelsDirTest.getAbsolutePath().toString() + "/%d.csv", 0, testSamples - 1));

        DataSetIterator testIter = new SequenceRecordReaderDataSetIterator(rrTestFeature, rrTestLabel, BATCH_SIZE, NUM_CLASSES, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);


        testIter.setPreProcessor(normalizer);


        while(testIter.hasNext())
        {
            DataSet dt = testIter.next();

            INDArray labels = dt.getLabels();

            int k = 0;
        }
        /*
        //check output label size with align end [NOTE: change the minibatch size to 1]

        DataSet data = testIter.next();

        System.out.println("data length: " + data.getFeatureMatrix().length());
        System.out.println("label length: " + data.getLabels().length());
        System.out.println(data.getLabelsMaskArray().toString());
        System.out.println(data.toString());

        System.out.println("********************************");
        System.out.println("********************************");
        */

        // ----- Configure the network -----
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(123)    //Random number generator seed for improved repeatability. Optional.
            .weightInit(WeightInit.XAVIER)
            .updater(new Nesterovs(0.005))
            .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)  //Not always required, but helps with this data set
            .gradientNormalizationThreshold(0.5)
            .list()
            .layer(0, new GravesLSTM.Builder()
                .activation(Activation.TANH)
                .nIn(1)
                .nOut(10)
                .build())
            .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                .activation(Activation.SOFTMAX)
                .nIn(10)
                .nOut(NUM_CLASSES)
                .build())
            .pretrain(false)
            .backprop(true)
            .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        net.setListeners(new ScoreIterationListener(20));   //Print the score (loss function value) every 20 iterations


        // ----- Train the network, evaluating the test set performance at each epoch -----
        int nEpochs = 20;
        String str = "Test set evaluation at epoch %d: Accuracy = %.2f, F1 = %.2f";
        for (int i = 0; i < nEpochs; i++) {
            net.fit(trainIter);

            trainIter.reset();
            testIter.reset();
        }

        //Evaluate on the test set:
        Evaluation evaluation = net.evaluate(testIter);

        System.out.println("Evaluation 1: \n" + evaluation.confusionToString());
        System.out.println(evaluation.stats());

        //Evaluate on the test set:
        testIter.reset();
        Evaluation eval2 = new Evaluation(NUM_CLASSES);
        while(testIter.hasNext())
        {
            DataSet dt = testIter.next();
            INDArray output = net.output(dt.getFeatureMatrix());

            eval2.evalTimeSeries(output, dt.getLabels(), dt.getLabelsMaskArray());
        }

        System.out.println("Evaluation 2: \n" + eval2.confusionToString());
        System.out.println(eval2.stats());


        testIter.reset();


        while(testIter.hasNext())
        {
            DataSet dt = testIter.next();

            INDArray feature = dt.getFeatureMatrix();
            for(int i = 0; i < BATCH_SIZE; ++i)
            {
                INDArray ft = feature.getRow(i);

                int[] predicted = net.predict(ft);
                int k = i + 1;

            }


        }


    }

    //This method downloads the data, and converts the "one time series per line" format into a suitable
    //CSV sequence format that DataVec (CsvSequenceRecordReader) and DL4J can read.
    private static void downloadUCIData() throws Exception
    {
        String url = "https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/synthetic_control.data";
        String data = IOUtils.toString(new URL(url));

        String[] lines = data.split("\n");

        //Create directories
        baseDir.mkdir();
        baseTrainDir.mkdir();
        featuresDirTrain.mkdir();
        labelsDirTrain.mkdir();
        baseTestDir.mkdir();
        featuresDirTest.mkdir();
        labelsDirTest.mkdir();

        int lineCount = 0;
        List<Pair<String, Integer>> contentAndLabels = new ArrayList<>();

        for (String line : lines) {
            String transposed = line.replaceAll(" +", "\n");

            //Labels: first 100 examples (lines) are label 0, second 100 examples are label 1, and so on
            contentAndLabels.add(new Pair<>(transposed, lineCount++ / 100));
        }

        //Randomize and do a train/test split:
        Collections.shuffle(contentAndLabels, new Random(12345));

        int trainCount = 0;
        int testCount = 0;
        for (Pair<String, Integer> p : contentAndLabels)
        {
            //Write output in a format we can read, in the appropriate locations
            File outPathFeatures;
            File outPathLabels;

            if (trainCount < trainSamples)
            {
                outPathFeatures = new File(featuresDirTrain, trainCount + ".csv");
                outPathLabels = new File(labelsDirTrain, trainCount + ".csv");
                trainCount++;
            }
            else {
                outPathFeatures = new File(featuresDirTest, testCount + ".csv");
                outPathLabels = new File(labelsDirTest, testCount + ".csv");
                testCount++;
            }

            FileUtils.writeStringToFile(outPathFeatures, p.getFirst());


            FileUtils.writeStringToFile(outPathLabels, p.getSecond().toString());
        }

        log.info("Download file completed.");

    }

}
