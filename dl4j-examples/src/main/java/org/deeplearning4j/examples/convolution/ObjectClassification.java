package org.deeplearning4j.examples.convolution;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;
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
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.lang.annotation.Native;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static java.lang.Math.toIntExact;
import static org.opencv.core.CvType.CV_8U;

public class ObjectClassification
{
    protected static final Logger log = LoggerFactory.getLogger(ObjectClassification.class);
    protected static int height = 100;
    protected static int width = 100;
    protected static int channels = 3;
    protected static int batchSize = 20;

    protected static long seed = 42;
    protected static Random rng = new Random(seed);
    protected static int epochs = 50;
    protected static double splitTrainTest = 0.8;
    protected static boolean save = false;

    protected static int maxPathsPerLabel = 65;

    protected double learningRate = 0.005;
    protected double dropOut = 0.5;
    protected List<String> labelArray = Arrays.asList(new String[]{"Dog", "Donut", "Earth"});

    private int numLabels;

    public void run() throws Exception {
        String savedPath = new ClassPathResource("objectclassification").getFile().toString();
        String saveAs = savedPath + "/objectModel.zip";

        MultiLayerNetwork network = null;

        if (new File(saveAs).exists() == false) {
            ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

            /**
             * Data Setup -> file path setup
             **/
            File mainPath = new File(System.getProperty("user.dir"), "dl4j-examples/src/main/resources/objectclassification");
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
            List<Pair<ImageTransform, Double>> pipeline = Arrays.asList(new Pair<>(flipTransform1, 0.9),
                    new Pair<>(flipTransform2, 0.8),
                    new Pair<>(warpTransform, 0.5));

            ImageTransform transform = new PipelineImageTransform(pipeline, shuffle);

            log.info("Build model....");

            /**
             * Set up network configuration
             */

            MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                    .seed(seed)
                    .weightInit(WeightInit.XAVIER)
                    .updater(new Nesterovs(learningRate, Nesterovs.DEFAULT_NESTEROV_MOMENTUM))
                    .l2(0.0004)
                    .list()
                    .layer(0, convInit("convInit", channels, 200, new int[]{11, 11}, new int[]{2, 2}, new int[]{3, 3}))
                    .layer(1, maxPool("pool1", new int[]{2, 2}))
                    .layer(2, conv("conv1", 150, new int[]{7, 7}, new int[]{2, 2}, new int[]{3, 3}, Activation.RELU))
                    .layer(3, maxPool("pool2", new int[]{2, 2}))
                    .layer(4, conv("conv2_1", 150, new int[]{5, 5}, new int[]{2, 2}, new int[]{3, 3}, Activation.IDENTITY))
                    .layer(5, conv("conv2_2", 125, new int[]{5, 5}, new int[]{2, 2}, new int[]{3, 3}, Activation.RELU))
                    .layer(6, maxPool("pool3", new int[]{2, 2}))
                    .layer(7, conv("conv3_1", 125, new int[]{3, 3}, new int[]{2, 2}, new int[]{3, 3}, Activation.IDENTITY))
                    .layer(8, conv("conv3_2", 100, new int[]{3, 3}, new int[]{2, 2}, new int[]{3, 3}, Activation.RELU))
                    .layer(9, maxPool("pool4", new int[]{2, 2}))
                    .layer(10, fullyConnected("fc1", 100, 0, dropOut))
                    .layer(11, fullyConnected("fc2", 80, 0, dropOut))
                    .layer(12, new OutputLayer.Builder()
                            .name("softmax")
                            .nOut(numLabels)
                            .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                            .activation(Activation.SOFTMAX)
                            .build())
                    .setInputType(InputType.convolutional(height, width, channels))
                    .build();

            /*
            MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                    .seed(seed)
                    .weightInit(WeightInit.XAVIER)
                    .updater(new Nesterovs(learningRate, Nesterovs.DEFAULT_NESTEROV_MOMENTUM))
                    .l2(0.0004)
                    .list()
                    .layer(0, convInit("convInit", channels, 120, new int[]{7, 7}, new int[]{2, 2}, new int[]{3, 3}))
                    .layer(1, conv("conv1", 120, new int[]{7, 7}, new int[]{2, 2}, new int[]{3, 3}))
                    .layer(2, maxPool("pool1", new int[]{2, 2}))
                    .layer(3, conv("conv2", 100, new int[]{5, 5}, new int[]{1, 1}, new int[]{3, 3}))
                    .layer(4, conv("conv2", 100, new int[]{5, 5}, new int[]{1, 1}, new int[]{3, 3}))
                    .layer(5, maxPool("pool2", new int[]{2, 2}))
                    .layer(6, conv("conv2", 80, new int[]{3, 3}, new int[]{1, 1}, new int[]{3, 3}))
                    .layer(7, conv("conv2", 80, new int[]{3, 3}, new int[]{1, 1}, new int[]{3, 3}))
                    .layer(8, maxPool("pool2", new int[]{2, 2}))
                    .layer(9, fullyConnected("fc1", 80, 0, dropOut))
                    .layer(10, fullyConnected("fc2", 80, 0, dropOut))
                    .layer(11, new OutputLayer.Builder()
                            .name("softmax")
                            .nOut(numLabels)
                            .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                            .activation(Activation.SOFTMAX)
                            .build())
                    .setInputType(InputType.convolutional(height, width, channels))
                    .build();
             */

            /**
             * Build model
             */
            log.info("Build model....");

            network = new MultiLayerNetwork(config);
            network.init();

            /**
             * Set up listener
             **/
            UIServer uiServer = UIServer.getInstance();
            StatsStorage statsStorage = new InMemoryStatsStorage();
            uiServer.attach(statsStorage);
            network.setListeners(new ScoreIterationListener(10), new StatsListener(statsStorage));

            /**
             * Data Setup -> normalization
             *  - how to normalize images and generate large dataset to train on
             **/
            DataNormalization scaler = new ImagePreProcessingScaler(0, 1);

            /**
             * Data Formatting
             **/
            ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
            DataSetIterator dataIter;

            //recordReader.initialize(trainData, transform); //train with transformations
            recordReader.initialize(trainData);//train without transformations

            dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);

            scaler.fit(dataIter);
            dataIter.setPreProcessor(scaler);

            /**
             * Start training
             **/
            log.info("Train model....");

            for (int i = 0; i < epochs; ++i) {
                network.fit(dataIter);
            }

            /**
             * Save network
             **/
            ModelSerializer.writeModel(network, saveAs, false);

            /**
             * Evaluate model
             **/
            log.info("Evaluate model....");
            recordReader.initialize(testData);
            dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
            scaler.fit(dataIter);
            dataIter.setPreProcessor(scaler);
            Evaluation eval = network.evaluate(dataIter);
            log.info(eval.stats());
        }
        else
        {
            /**
             * Load saved model
             **/
            log.info("Saved model exist. Load network");
            network = ModelSerializer.restoreMultiLayerNetwork(saveAs);

            File singleImagePath = new ClassPathResource("samoyed.jpg").getFile();

            /**
             * Single Image Loader
             **/
            NativeImageLoader loader = new NativeImageLoader(height, width, channels);

            INDArray image = loader.asMatrix(singleImagePath);

            ImagePreProcessingScaler preprocessor = new ImagePreProcessingScaler(0, 1);
            preprocessor.preProcess(image);

            /**
             * Test single image
             **/

            String labelImage = labelArray.get(network.predict(image)[0]);
            log.info("Probabilities: " + network.output(image).toString());
            log.info("Label of single image: " + labelImage);

            /**For display
             *
             */

            NativeImageLoader showLoader = new NativeImageLoader();//200, 200, 3);//height, width, channels);

            INDArray indImage = showLoader.asMatrix(singleImagePath);
            opencv_core.Mat matImage = loader.asMat(indImage);
            matImage.convertTo(matImage, CV_8U);
            OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();

            Frame frameImage = converter.convert(matImage);

            CanvasFrame canvasFrame = new CanvasFrame(labelImage);
            canvasFrame.setCanvasSize(frameImage.imageWidth, frameImage.imageHeight);//matImage.cols(), matImage.rows());
            canvasFrame.showImage(frameImage);
            canvasFrame.waitKey();
            canvasFrame.dispose();

        }

    }

    public static void main(String[] args) throws Exception
    {
        ObjectClassification classifier = new ObjectClassification();

        classifier.run();
    }

    private ConvolutionLayer convInit(String name, int in, int out, int[] kernel, int[] stride, int[] pad)
    {
        return new ConvolutionLayer.Builder()
                .kernelSize(kernel)
                .stride(stride)
                .padding(pad)
                .name(name)
                .nIn(in)
                .nOut(out)
                .biasInit(0)
                .activation(Activation.IDENTITY)
                .build();

    }

    private ConvolutionLayer conv(String name, int out, int[] kernel,int[] stride, int[] pad, Activation activationFunc)
    {
        return new ConvolutionLayer.Builder()
                .kernelSize(kernel)
                .stride(stride)
                .padding(pad)
                .name(name)
                .nOut(out)
                .activation(activationFunc)
                .build();

    }

    private SubsamplingLayer maxPool(String name, int[] kernel)
    {
        return new SubsamplingLayer.Builder()
                .poolingType(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(kernel)
                .stride(new int[]{2,2})
                .name(name)
                .build();
    }

    private DenseLayer fullyConnected(String name, int out, double bias, double dropOut)
    {
        return new DenseLayer.Builder()
                .name(name)
                .nOut(out)
                .biasInit(bias)
                .dropOut(dropOut)
                .build();
    }
}


//how to know its correct label
//reorganize the code for training