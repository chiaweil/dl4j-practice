package org.deeplearning4j.solutions.feedforward.mnist;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.IAMax;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Random;

/**
 *
 *  This code example is featured in this youtube video
 *
 *  https://www.youtube.com/watch?v=zrTSs715Ylo
 *
 ** This differs slightly from the Video Example,
 * The Video example had the data already downloaded
 * This example includes code that downloads the data
 *
 * Data SOurce
 *  wget http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz
 *  followed by tar xzvf mnist_png.tar.gz
 *
 *  OR
 *  git clone https://github.com/myleott/mnist_png.git
 *  cd mnist_png
 *  tar xvf mnist_png.tar.gz
 *
 *
 *  This examples builds on the MnistImagePipelineExample
 *  by Saving the Trained Network
 *
 */

public class MnistImageSave {
    /**
     * Data URL for downloading
     */
    private static Logger log = LoggerFactory.getLogger(MnistImageSave.class);

    public static void main(String[] args) throws Exception {

        // image information
        // 28 * 28 grayscale
        // grayscale implies single channel
        int height = 28;
        int width = 28;
        int channels = 1;
        int rngseed = 123;

        Random randNumGen = new Random(rngseed);

        int batchSize = 500;
        int outputNum = 10;
        int numEpochs = 10;

        File locationToSave = new File(new ClassPathResource("mnist_png").getFile().getAbsolutePath().toString() + "/trained_mnist_model.zip");

        //Define the File Paths
        File trainData = new ClassPathResource("mnist_png/training").getFile();
        File testingData = new ClassPathResource("mnist_png/testing").getFile();

        FileSplit fileSplitTrain = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
        FileSplit fileSplitTest = new FileSplit(testingData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);

        //RecordReader
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator(); //Extract the parent path as the image label

        ImageRecordReader rrTrain = new ImageRecordReader(height, width, channels, labelMaker);
        rrTrain.initialize(fileSplitTrain);

        ImageRecordReader rrTest = new ImageRecordReader(height, width, channels, labelMaker);
        rrTest.initialize(fileSplitTest);

        //DataSet Iterator
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rrTrain, batchSize, 1, outputNum);
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest, batchSize, 1, outputNum);

        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(trainIter);

        trainIter.setPreProcessor(scaler);
        testIter.setPreProcessor(scaler);

        MultiLayerNetwork model;

        if (locationToSave.exists() == false) {

            System.out.println("Build model");

            //Build neural network configuration
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(rngseed)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .updater(new Nesterovs())
                    .l2(1e-4)
                    .list()
                    .layer(0, new DenseLayer.Builder()
                            .nIn(height * width)
                            .nOut(200)
                            .activation(Activation.RELU)
                            .weightInit(WeightInit.XAVIER)
                            .build())
                    .layer(1, new DenseLayer.Builder()
                            .nIn(200)
                            .nOut(100)
                            .activation(Activation.RELU)
                            .weightInit(WeightInit.XAVIER)
                            .build())
                    .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                            .nIn(100)
                            .nOut(outputNum)
                            .activation(Activation.SOFTMAX)
                            .weightInit(WeightInit.XAVIER)
                            .build())
                    .setInputType(InputType.convolutional(height, width, channels))
                    .build();

            model = new MultiLayerNetwork(conf);
            model.init();

            model.setListeners(new ScoreIterationListener(10));

            UIServer uiServer = UIServer.getInstance();
            StatsStorage statsStorage = new InMemoryStatsStorage();
            model.setListeners(new StatsListener(statsStorage));
            uiServer.attach(statsStorage);


            log.info("*****TRAIN MODEL********");
            int evalCycle = 3;
            for (int i = 0; i < numEpochs; i++) {
                model.fit(trainIter);

                if ((i % evalCycle) == 0) {
                    System.out.println("Epoch " + i);
                    Evaluation eval = new Evaluation(outputNum);

                    while (testIter.hasNext()) {
                        DataSet dt = testIter.next();

                        INDArray predictedLabels = model.output(dt.getFeatureMatrix());
                        eval.eval(dt.getLabels(), predictedLabels);
                    }

                    System.out.println(eval.stats());
                    testIter.reset();
                }
            }


            log.info("******SAVE TRAINED MODEL******");
            // Where to save model

            log.info(locationToSave.toString());

            // boolean save Updater
            boolean saveUpdater = false;

            // ModelSerializer needs modelname, saveUpdater, Location
            ModelSerializer.writeModel(model, locationToSave, saveUpdater);
        }
        else
        {
            System.out.println("Trained Model exist. Proceed with Evaluation");

            model = ModelSerializer.restoreMultiLayerNetwork(locationToSave);
        }

        System.out.println("Final Evaluation");



        Evaluation eval = new Evaluation(outputNum);

        while(testIter.hasNext())
        {
            DataSet dt = testIter.next();

            INDArray setPredicted = model.output(dt.getFeatureMatrix());
            INDArray setLabels = dt.getLabels();

            eval.eval(setPredicted, setLabels);
            /*
            for(int i = 0; i < batchSize; ++i)
            {
                INDArray predicted = setPredicted.getRow(i);
                int y_hat = Nd4j.getExecutioner().execAndReturn(new IAMax(predicted)).getFinalResult();

                INDArray labels = setLabels.getRow(i);
                int y = Nd4j.getExecutioner().execAndReturn(new IAMax(labels)).getFinalResult();

                if(y_hat != y)
                {
                    System.out.println("Wrong Label");
                }
                System.out.format("y_hat: %d y: %d\n", y_hat, y);

            }
            */
        }

        log.info(eval.stats());

    }
}