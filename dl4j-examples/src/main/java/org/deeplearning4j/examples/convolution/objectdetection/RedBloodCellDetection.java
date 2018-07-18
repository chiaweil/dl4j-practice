package org.deeplearning4j.examples.convolution.objectdetection;

import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.datavec.image.recordreader.objdetect.impl.VocLabelProvider;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.model.TinyYOLO;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.deeplearning4j.nn.weights.WeightInit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.Iterator;
import java.util.Random;

public class RedBloodCellDetection
{
    private static final Logger log = LoggerFactory.getLogger(RedBloodCellDetection.class);

    public static void main(String[] args) throws Exception
    {
        //parameters matching the pretrained TinyYOLO model
        int width = 416;
        int height = 416;
        int channels = 3;
        int gridWidth = 13;
        int gridHeight = 13;

        //number class(digits) for the SVHN datasets
        int classes = 1;

        //parameters for the Yolo2OutputLayer
        int nBoxes = 5;
        double lambdaNoObj = 0.5;
        double lambdaCoord = 5.0;
        double[][] priorBoxes = {{2, 2}, {2, 2}, {2, 2}, {2, 2}, {2, 2}};
        double detectionThreshold = 0.3;

        //parameters for the training phase
        int batchSize = 10;
        int nEpochs = 50;
        double learningRate = 1e-3;
        double lrMomentum = 0.9;

        int seed = 123;
        Random rng = new Random(seed);

        log.info("Load data...");

        //Randomizes the order of paths in an array
        RandomPathFilter pathFilter = new RandomPathFilter(rng)
        {
            @Override
            protected boolean accept(String name) {
                name = name.replace("/JPEGImages/", "/Annotations/").replace(".jpg", ".xml");
                try
                {
                    boolean isXMLExist = new File(new URI(name)).exists();

                    return isXMLExist;

                } catch (URISyntaxException ex) {
                    throw new RuntimeException(ex);
                }
            }
        };

        String dataDir = new ClassPathResource("RedBloodCell/cosmicad/").getFile().getAbsolutePath();

        File imageDir = new File(dataDir, "JPEGImages");

        //From a folder, shuffle images and split them to train and test data
        FileSplit fileSplit = new FileSplit(imageDir, NativeImageLoader.ALLOWED_FORMATS, rng);

        //Randomized the order of paths in an array
        //RandomPathFilter pathFilter = new RandomPathFilter(rng);

        InputSplit[] data = fileSplit.sample(pathFilter, 0.8, 0.2);

        InputSplit trainData = data[0];
        InputSplit testData = data[1];


        ObjectDetectionRecordReader rrTrain = new ObjectDetectionRecordReader(height, width, channels,
                gridHeight, gridWidth, new VocLabelProvider(dataDir));
        rrTrain.initialize(trainData);

        ObjectDetectionRecordReader rrTest = new ObjectDetectionRecordReader(height, width, channels,
                gridHeight, gridWidth, new VocLabelProvider(dataDir));
        rrTest.initialize(testData);


        // ObjectDetectionRecordReader performs regression, so we need to specify it here
        RecordReaderDataSetIterator iterTrain = new RecordReaderDataSetIterator(rrTrain, batchSize, 1, 1, true);
        iterTrain.setPreProcessor(new ImagePreProcessingScaler(0, 1));


        RecordReaderDataSetIterator iterTest = new RecordReaderDataSetIterator(rrTest, batchSize, 1, 1, true);
        iterTest.setPreProcessor(new ImagePreProcessingScaler(0, 1));


        ComputationGraph model;

        String modelFileName = "/model_rbc.zip";

        File modelPath = new File(dataDir, modelFileName);
        
        if(modelPath.exists())
        {
            log.info("Load model from " + modelFileName);

            model = ModelSerializer.restoreComputationGraph(modelFileName);
        }
        else
        {
            log.info("Build model...");

            ComputationGraph pretrained = (ComputationGraph) TinyYOLO.builder().build().initPretrained();
            INDArray priors = Nd4j.create(priorBoxes);


            FineTuneConfiguration fineTuneConfig = new FineTuneConfiguration.Builder()
                    .seed(seed)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                    .gradientNormalizationThreshold(0.5)
                    .updater(new Adam.Builder()
                            .learningRate(learningRate)
                            .build())
                    .activation(Activation.IDENTITY)
                    .trainingWorkspaceMode(WorkspaceMode.NONE)
                    .inferenceWorkspaceMode(WorkspaceMode.NONE)
                    .build();

            model = new TransferLearning.GraphBuilder(pretrained)
                    .fineTuneConfiguration(fineTuneConfig)
                    .removeVertexKeepConnections("conv2d_9")
                    .addLayer("convolution2d_9",
                            new ConvolutionLayer.Builder(1,1)
                                    .nIn(1024)
                                    .nOut(nBoxes * (5 + classes))
                                    .stride(1, 1)
                                    .weightInit(WeightInit.UNIFORM)
                                    .convolutionMode(ConvolutionMode.Same)
                                    .hasBias(false)
                                    .activation(Activation.IDENTITY)
                                    .build(),
                            "leaky_re_lu_8")
                    .addLayer("outputs",
                            new Yolo2OutputLayer.Builder()
                                    .lambbaNoObj(lambdaNoObj)
                                    .lambdaCoord(lambdaCoord)
                                    .boundingBoxPriors(priors)
                                    .build(),
                            "convolution2d_9")
                    .setOutputs("outputs")
                    .build();

            System.out.println(model.summary(InputType.convolutional(height, width, channels)));

            log.info("Train model...");

            model.setListeners(new ScoreIterationListener(10));

            for(int i = 0; i < nEpochs; ++i)
            {
                model.fit(iterTrain);
                log.info("*** Completed epoch {} ***", i);
            }

            ModelSerializer.writeModel(model, modelFileName, true);


        }


    }
}

//lambdaNoObj??
//lambdaCoord??
//nBoxes * (5 + nClasses)