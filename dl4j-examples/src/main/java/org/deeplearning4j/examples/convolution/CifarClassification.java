package org.deeplearning4j.examples.convolution;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator;
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
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

/**
 * Train CNN using cifar dataset
 * Cifar dataset has total 50000 training data, 10000 testing data
 */
public class CifarClassification
{
    public static Logger log = LoggerFactory.getLogger(CifarClassification.class);

    public static void main(String[] args) throws Exception
    {
        CifarClassification cifarCNN = new CifarClassification();

        cifarCNN.run();
    }

    public void run() throws Exception
    {
        int height = 32;
        int width = 32;
        int channels = 3;

        int totalLabels = 10;
        int batchSize = 250;
        int totalSamples = 5000; //num of samples use for this training
        boolean train = true;

        int seed = 123;
        double learningRate = 0.001;
        double dropOut = 0.5;
        int epoch = 50;

        /**
         * Data download and set up
         */
        CifarDataSetIterator dataIterTrain = new CifarDataSetIterator(batchSize, totalSamples, train);
        CifarDataSetIterator dataIterEval = new CifarDataSetIterator(batchSize, totalSamples, false);

        log.info(dataIterTrain.getLabels().toString());

        /**
         * Set up network configuration
         */

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
                        .nOut(totalLabels)
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
                .layer(4, maxPool("pool2", new int[]{2, 2}))
                .layer(5, conv("conv2", 100, new int[]{3, 3}, new int[]{1, 1}, new int[]{3, 3}))
                .layer(6, maxPool("pool2", new int[]{2, 2}))
                .layer(7, fullyConnected("fc1", 80, 0, dropOut))
                .layer(8, fullyConnected("fc2", 50, 0, dropOut))
                .layer(9, new OutputLayer.Builder()
                        .name("softmax")
                        .nOut(totalLabels)
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

        MultiLayerNetwork network = new MultiLayerNetwork(config);
        network.init();

        /**
         * Set up listener
         **/
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        network.setListeners(new ScoreIterationListener(10));//, new StatsListener(statsStorage));



        /**
         * Train network
         **/
        log.info("Start training....");
        network.fit(dataIterTrain, epoch);

        /**
         * Save network
         **/
        String saveAs = "cifarNetwork.zip";
        ModelSerializer.writeModel(network, saveAs, false);

        /**
         * Evaluate model
         **/
        log.info("Final evaluation on evaluation data....");
        Evaluation eval = network.evaluate(dataIterEval);
        log.info(eval.stats());



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

    private ConvolutionLayer conv(String name, int out, int[] kernel,int[] stride, int[] pad)
    {
        return new ConvolutionLayer.Builder()
                .kernelSize(kernel)
                .stride(stride)
                .padding(pad)
                .name(name)
                .nOut(out)
                .activation(Activation.RELU)
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



//save the model
//use more data
//considering changing to easier data (u gonna do it anyways)
//try with AlexNet
//why ui interface not working
//build another data