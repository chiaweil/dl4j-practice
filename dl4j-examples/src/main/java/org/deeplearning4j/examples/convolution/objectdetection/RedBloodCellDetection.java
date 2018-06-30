package org.deeplearning4j.examples.convolution.objectdetection;

import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.util.ClassPathResource;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.datavec.image.recordreader.objdetect.impl.VocLabelProvider;
import org.deeplearning4j.api.storage.StatsStorage;
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
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.model.TinyYOLO;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.Random;

/**
 * Example transfer learning from a Tiny YOLO model pretrained on ImageNet and Pascal VOC
 * to perform object detection with bounding boxes on images of red blood cells.
 * <p>
 * References: <br>
 * - YOLO: Real-Time Object Detection: https://pjreddie.com/darknet/yolo/ <br>
 * - Images of red blood cells: https://github.com/cosmicad/dataset <br>
 * <p>
 * Please note, cuDNN should be used to obtain reasonable performance: https://deeplearning4j.org/cudnn
 *
 * @author saudet
 */

public class RedBloodCellDetection
{
    private static final Logger log = LoggerFactory.getLogger(RedBloodCellDetection.class);


    public static void main(String[] args) throws Exception
    {
        // parameters matching the pretrained TinyYOLO model
        int width = 416;
        int height = 416;
        int nChannels = 3;
        int gridWidth = 13;
        int gridHeight = 13;

        // number classes for the red blood cells (RBC)
        int nClasses = 1;

        // parameters for the Yolo2OutputLayer
        int nBoxes = 5;
        double lambdaNoObj = 0.5;
        double lambdaCoord = 5.0;
        double[][] priorBoxes = {{2, 2}, {2, 2}, {2, 2}, {2, 2}, {2, 2}};
        double detectionThreshold = 0.3;

        // parameters for the training phase
        int batchSize = 10;
        int nEpochs = 50;
        double learningRate = 1e-3;
        double lrMomentum = 0.9;

        int seed = 123;
        Random rng = new Random(seed);

        String dataDir = new ClassPathResource("/RedBloodCell/cosmicad/").getFile().getAbsolutePath();
        File imageDir = new File(dataDir, "JPEGImages");

        log.info("Load data...");

        /*
        if(imageDir.exists())
        {
            log.info("Dataset ready...");
        }
        else
        {
            log.info("Dataset not ready...");
        }

        RandomPathFilter pathFilter = new RandomPathFilter(rng) {
            @Override
            protected boolean accept(String name) {
                name = name.replace("/JPEGImages/", "/Annotations/").replace(".jpg", ".xml");

                try
                {
                    return new File(new URI(name)).exists();
                }
                catch (URISyntaxException ex)
                {
                    throw new RuntimeException(ex);
                }
            }
        };


        InputSplit[] data = new FileSplit(imageDir, NativeImageLoader.ALLOWED_FORMATS, rng).sample(pathFilter, 0.8, 0.2);
        InputSplit trainData = data[0];
        InputSplit testData = data[1];

        ObjectDetectionRecordReader recordReaderTrain = new ObjectDetectionRecordReader(height, width, nChannels,
            gridHeight, gridWidth, new VocLabelProvider(dataDir));
        recordReaderTrain.initialize(trainData);

        ObjectDetectionRecordReader recordReaderTest = new ObjectDetectionRecordReader(height, width, nChannels,
            gridHeight, gridWidth, new VocLabelProvider(dataDir));
        recordReaderTest.initialize(testData);

        // ObjectDetectionRecordReader performs regression, so we need to specify it here
        RecordReaderDataSetIterator train = new RecordReaderDataSetIterator(recordReaderTrain, batchSize, 1, 1, true);
        train.setPreProcessor(new ImagePreProcessingScaler(0, 1));

        RecordReaderDataSetIterator test = new RecordReaderDataSetIterator(recordReaderTest, 1, 1, 1, true);
        test.setPreProcessor(new ImagePreProcessingScaler(0, 1));

        ComputationGraph model;
        String modelFilename = "model_rbc.zip";

        if (new File(modelFilename).exists())
        {
            log.info("Load model...");

            model = ModelSerializer.restoreComputationGraph(modelFilename);

        }
        else {
            log.info("Build model...");


            FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(1.0)
                .updater(new Adam.Builder().learningRate(learningRate).build())
                .activation(Activation.IDENTITY)
                .trainingWorkspaceMode(WorkspaceMode.NONE)
                .inferenceWorkspaceMode(WorkspaceMode.NONE)
                .build();


            ComputationGraph pretrained = (ComputationGraph) new TinyYOLO().initPretrained();


            INDArray priors = Nd4j.create(priorBoxes);

            model = new TransferLearning.GraphBuilder(pretrained)
                .fineTuneConfiguration(fineTuneConf)
                .removeVertexKeepConnections("conv2d_9")
                .addLayer("convolution2d_9",
                    new ConvolutionLayer.Builder(1, 1)
                        .nIn(1024)
                        .nOut(nBoxes * (5 + nClasses))
                        .stride(1, 1)
                        .convolutionMode(ConvolutionMode.Same)
                        .weightInit(WeightInit.UNIFORM)
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

            System.out.println(model.summary(InputType.convolutional(height, width, nChannels)));

            log.info("Train model...");

            UIServer server = UIServer.getInstance();
            StatsStorage storage = new InMemoryStatsStorage();
            server.attach(storage);

            model.setListeners(new StatsListener(storage,10));

            for(int i = 0; i < nEpochs; ++i)
            {
                System.out.println("Epoch: " + i);
                train.reset();

                while(train.hasNext())
                {

                    model.fit(train.next());
                }
            }

            ModelSerializer.writeModel(model, modelFilename, true);
            System.out.println("Model written at: " + modelFilename);
        }
            */


    }
}


//priorBoxes
//fineTuneConfig details
