package org.deeplearning4j.examples.convolution;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.PipelineImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.StepSchedule;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static java.lang.Math.toIntExact;

/**
 * Animal Classification
 *
 * Example classification of photos from 4 different animals (bear, duck, deer, turtle).
 *
 * References:
 *  - U.S. Fish and Wildlife Service (animal sample dataset): http://digitalmedia.fws.gov/cdm/
 *  - Tiny ImageNet Classification with CNN: http://cs231n.stanford.edu/reports/2015/pdfs/leonyao_final.pdf
 *
 * CHALLENGE: Current setup gets low score results. Can you improve the scores? Some approaches:
 *  - Add additional images to the dataset
 *  - Apply more transforms to dataset
 *  - Increase epochs
 *  - Try different model configurations
 *  - Tune by adjusting learning rate, updaters, activation & loss functions, regularization, ...
 *
 */

public class AnimalsClassification
{
    protected static final Logger log = LoggerFactory.getLogger(AnimalsClassification.class);
    protected static int height = 100;
    protected static int width = 100;
    protected static int channels = 3;
    protected static int batchSize = 20;

    protected static long seed = 42;
    protected static Random rng = new Random(seed);
    protected static int epochs = 50;
    protected static double splitTrainTest = 0.8;
    protected static boolean save = false;
    protected static int maxPathsPerLabel = 18;

    protected static String modelType = "AlexNet"; // LeNet, AlexNet or Custom but you need to fill it out
    private int numLabels;

    public void run() throws Exception
    {
        log.info("Load data....");

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        /**
         * Data Setup -> file path setup
         **/
        File mainPath = new File(System.getProperty("user.dir"), "dl4j-examples/src/main/resources/animals");
        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, rng);

        int numExamples = toIntExact(fileSplit.length());

        numLabels = fileSplit.getRootDir().listFiles(File::isDirectory).length; //This only works if your root is clean: only label subdirs.

        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, labelMaker, numExamples, numLabels, maxPathsPerLabel);

        /**
         * Data Setup -> train test split
         *  - inputSplit = define train and test split
         **/
        InputSplit[] inputSplit = fileSplit.sample(pathFilter, splitTrainTest, 1 - splitTrainTest);
        InputSplit trainData = inputSplit[0];
        InputSplit testData = inputSplit[1];

        /**
         * Data Setup -> transformation
         *  - Transform = how to tranform images and generate large dataset to train on
         **/
        ImageTransform flipTransform1 = new FlipImageTransform(rng);
        ImageTransform flipTransform2 = new FlipImageTransform(new Random(123));
        ImageTransform warpTransform = new WarpImageTransform(rng, 42);

        boolean shuffle = false;
        List< Pair<ImageTransform,Double> > pipeline = Arrays.asList(new Pair<>(flipTransform1,0.9),
                                                                   new Pair<>(flipTransform2,0.8),
                                                                   new Pair<>(warpTransform,0.5));

        ImageTransform transform = new PipelineImageTransform(pipeline, shuffle);

        log.info("Build model....");

        // Uncomment below to try AlexNet. Note change height and width to at least 100
        // MultiLayerNetwork network = new AlexNet(height, width, channels, numLabels, seed, iterations).init();

        MultiLayerNetwork network;

        switch (modelType)
        {
            case "LeNet":
                network = lenetModel();
                break;
            case "AlexNet":
                network = alexnetModel();
                break;
            case "custom":
                network = null;//customModel();
                break;
            default:
                throw new InvalidInputTypeException("Incorrect model provided.");
        }

        network.init();

        /**
         * Set up listener
         **/
        //UIServer uiServer = UIServer.getInstance();
        //StatsStorage statsStorage = new InMemoryStatsStorage();
        //uiServer.attach(statsStorage);
        network.setListeners(new ScoreIterationListener(10));//, new StatsListener(statsStorage));

        /**
         * Data Setup -> normalization
         *  - how to normalize images and generate large dataset to train on
         **/
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);

        /**
         * Data Formatting
         **/
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);

        //recordReader.next().get(1).toString() // to get labels
        DataSetIterator dataIter;

        recordReader.initialize(trainData, transform); //train with transformations
        //recordReader.initialize(trainData);//train without transformations

        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);

        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);

        /**
         * Start training
         **/
        log.info("Train model....");
        network.fit(dataIter, epochs);

        /**
         * Evaluate model
         **/
        log.info("Evaluate model....");
        recordReader.initialize(testData);
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);
        Evaluation eval = network.evaluate(dataIter);
        log.info(eval.stats(true));
    }

    public static void main(String[] args) throws Exception
    {
        AnimalsClassification classifier = new AnimalsClassification();

        classifier.run();
    }



    public MultiLayerNetwork alexnetModel()
    {
        /**
         * AlexNet model interpretation based on the original paper ImageNet Classification with Deep Convolutional Neural Networks
         * and the imagenetExample code referenced.
         * http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
         **/

        double nonZeroBias = 1;
        double dropOut = 0.5;

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.DISTRIBUTION)
                .dist(new NormalDistribution(0.0, 0.01))
                .activation(Activation.RELU)
                .updater(new Nesterovs(new StepSchedule(ScheduleType.ITERATION, 1e-2, 0.1, 100000), 0.9))
                .biasUpdater(new Nesterovs(new StepSchedule(ScheduleType.ITERATION, 2e-2, 0.1, 100000), 0.9))
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                .l2(5 * 1e-4)
                .list()
                .layer(0, convInit("cnn1", channels, 96, new int[]{11, 11}, new int[]{4,4}, new int[]{3,3}, 0))
                .layer(1, new LocalResponseNormalization.Builder().name("lrn1").build())
                .layer(2, maxPool("maxpool1", new int[]{3,3}))
                .layer(3, conv5x5("cnn2", 256, new int[]{1,1}, new int[]{2,2}, nonZeroBias))
                .layer(4, new LocalResponseNormalization.Builder().name("lrn2").build())
                .layer(5, maxPool("maxpool2", new int[]{3,3}))
                .layer(6, conv3x3("cnn3", 384, 0))
                .layer(7,conv3x3("cnn4", 384, nonZeroBias))
                .layer(8,conv3x3("cnn5", 256, nonZeroBias))
                .layer(9, maxPool("maxpool3", new int[]{3,3}))
                .layer(10, fullyConnected("ffn1", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
                .layer(11, fullyConnected("ffn2", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
                .layer(12, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backprop(true)
                .pretrain(false)
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        return new MultiLayerNetwork(config);
    }

    public MultiLayerNetwork lenetModel() {
        /**
         * Revisde Lenet Model approach developed by ramgo2 achieves slightly above random
         * Reference: https://gist.github.com/ramgo2/833f12e92359a2da9e5c2fb6333351c5
         **/

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .l2(0.005)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(0.0001, 0.9))
                .list()
                .layer(0, convInit("cnn1", channels, 50, new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0}, 0))
                .layer(1, maxPool("maxpool1", new int[]{2, 2}))
                .layer(2, conv5x5("cnn2", 100, new int[]{5, 5}, new int[]{1, 1}, 0))
                .layer(3, maxPool("maxpool2", new int[]{2, 2}))
                .layer(4, new DenseLayer.Builder().nOut(500).build())
                .layer(5, new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backprop(true).pretrain(false)
                .setInputType(InputType.convolutional(height, width, channels))
                .build();


        return new MultiLayerNetwork(config);
    }


    public static MultiLayerNetwork customModel() {
        /**
         * Use this method to build your own custom model.
         **/
        return null;
    }

    private ConvolutionLayer convInit(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias)
    {
        return new ConvolutionLayer.Builder()
                .kernelSize(kernel)
                .padding(pad)
                .stride(stride)
                .name(name)
                .nIn(in)
                .nOut(out)
                .biasInit(bias)
                .build();

    }

    private ConvolutionLayer conv5x5(String name, int out, int[] stride, int[] pad, double bias)
    {
        return new ConvolutionLayer.Builder()
                .kernelSize(new int[]{5,5})
                .stride(stride)
                .padding(pad)
                .name(name)
                .nOut(out)
                .biasInit(bias)
                .build();
    }

    private ConvolutionLayer conv3x3(String name, int out, double bias)
    {
        return new ConvolutionLayer.Builder()
                .kernelSize(new int[]{3,3})
                .stride(new int[] {1,1})
                .padding(new int[] {1,1})
                .name(name)
                .nOut(out)
                .biasInit(bias)
                .build();
    }

    private SubsamplingLayer maxPool(String name, int[] kernel)
    {
        return new SubsamplingLayer.Builder()
                .kernelSize(kernel)
                .stride(new int[]{2,2})
                .name(name)
                .build();
    }

    private DenseLayer fullyConnected(String name, int out, double bias, double dropOut, Distribution dist)
    {
        return new DenseLayer.Builder()
                .name(name)
                .nOut(out)
                .biasInit(bias)
                .dropOut(dropOut)
                .dist(dist)
                .build();
    }
}


//will use 1 channel better??
//understand lenet configuration
//prepare another sample set
