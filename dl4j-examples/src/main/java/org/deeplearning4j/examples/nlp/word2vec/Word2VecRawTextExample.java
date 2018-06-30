package org.deeplearning4j.examples.nlp.word2vec;


import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.ui.api.UIServer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.ArrayList;
import java.util.Collection;

/**
 * Created by agibsonccc on 10/9/14.
 * <p>
 * Neural net that processes text into wordvectors. See below url for an in-depth explanation.
 * https://deeplearning4j.org/word2vec.html
 * <p>
 * Here are Deeplearning4j’s natural-language processing components:
 * <p>
 * SentenceIterator/DocumentIterator: Used to iterate over a dataset. A SentenceIterator returns strings and a DocumentIterator works with inputstreams.
 * Tokenizer/TokenizerFactory: Used in tokenizing the text. In NLP terms, a sentence is represented as a series of tokens. A TokenizerFactory creates an instance of a tokenizer for a “sentence.”
 * VocabCache: Used for tracking metadata including word counts, document occurrences, the set of tokens (not vocab in this case, but rather tokens that have occurred), vocab (the features included in both bag of words as well as the word vector lookup table)
 * Inverted Index: Stores metadata about where words occurred. Can be used for understanding the dataset. A Lucene index with the Lucene implementation[1] is automatically created.
 */

public class Word2VecRawTextExample {
    private static Logger log = LoggerFactory.getLogger(Word2VecRawTextExample.class);


    public static void main(String[] args) throws Exception {
        // Gets Path to Text file
        File filePath = new ClassPathResource("raw_sentences.txt").getFile();

        log.info("Load & Vectorize Sentences...");

        //U// Strip white space before and after for each line
        SentenceIterator iter = new LineSentenceIterator(filePath);

        //Split on white spaces in the line to get word
        //To tokenize a text is to break it up into its atomic units, creating a new token each time you hit a white space, for example.
        TokenizerFactory tokenizer = new DefaultTokenizerFactory();
        /*
            CommonPreprocessor will apply the following regex to each token: [\d\.:,"'\(\)\[\]|/?!;]+
            So, effectively all numbers, punctuation symbols and some special symbols are stripped off.
            Additionally it forces lower case for all tokens.
        */
        tokenizer.setTokenPreProcessor(new CommonPreprocessor());

        log.info("Building model...");

        Word2Vec vec = new Word2Vec.Builder()
            .minWordFrequency(5) //minimum number of times a word must appear in the corpus. Here, if it appears less than 5 times, it is not learned.
            .iterations(1) //this is the number of times you allow the net to update its coefficients for one batch of the data. Too few iterations mean it may not have time to learn all it can; too many will make the net’s training longer.
            .layerSize(100) //specifies the number of features in the word vector. This is equal to the number of dimensions in the featurespace. Words represented by 500 features become points in a 500-dimensional space.
            .seed(123)
            .windowSize(5) //how many words to affect it
            .iterate(iter) // tells the net what batch of the dataset it’s training on.
            .tokenizerFactory(tokenizer) //feeds it the words from the current batch.
            .build();


        //Set server
        UIServer server = UIServer.getInstance();
        System.out.println("Started on port " + server.getPort());

        log.info("Fitting Word2Vec model....");

        vec.fit();


        log.info("Writing word vectors to text file....");
        // Write word vectors
        WordVectorSerializer.writeWordVectors(vec, "wordsOutput.txt");

        String targetWord = "night";

        //difference between wordsNearest and wordsNearestSum is the latter return the own word
        getNearestWord(vec, targetWord, 10, log);
        getNearestWordSum(vec, targetWord, 10, log);


    }

    private static void getNearestWord(Word2Vec vec, String targetWord, int numberOfWords, Logger log) {

        // Prints out the closest 10 words to "day". An example on what to do with these Word Vectors.
        log.info("*********GetNearestWords*********");

        //Collection represents a single unit of objects i.e. a group.
        Collection<String> list = vec.wordsNearest(targetWord, numberOfWords);
        log.info("10 Words closest to 'day': {}", list);

        //Printing out the distance between the target word and each predicted near words
        ArrayList<Double> arrayDist = new ArrayList<>();

        for (String item : list) {
            double cosDis = vec.similarity(targetWord, item);
            arrayDist.add(cosDis);
        }

        log.info("Distance of words closest to 'day': {}", arrayDist);

    }

    private static void getNearestWordSum(Word2Vec vec, String targetWord, int numberOfWords, Logger log) {

        // Prints out the closest 10 words to "day". An example on what to do with these Word Vectors.
        log.info("*********GetNearestWordsSum*********");

        //return similar words which clustered semantically
        Collection<String> list = vec.wordsNearestSum(targetWord, numberOfWords);
        log.info("10 Words closest to 'day': {}", list);

        ArrayList<Double> arrayDist = new ArrayList<>();

        for (String item : list) {
            //return the cosine similarity of the two words
            double cosDis = vec.similarity(targetWord, item);
            arrayDist.add(cosDis);
        }

        log.info("Distance of words closest to 'day': {}", arrayDist);

    }


}
