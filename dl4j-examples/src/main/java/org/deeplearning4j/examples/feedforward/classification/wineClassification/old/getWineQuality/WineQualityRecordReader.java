package org.deeplearning4j.examples.feedforward.classification.wineClassification.old.getWineQuality;

import org.apache.commons.io.IOUtils;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.deeplearning4j.examples.feedforward.classification.wineClassification.old.getWineType.WineTypeRecordReader;

import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.*;
import java.util.regex.Pattern;


public class WineQualityRecordReader extends WineTypeRecordReader
{
    protected double ratioData;
    protected boolean startFromBeginning;


    public WineQualityRecordReader(int skipNumLines, char delimiter, double ratioData, boolean startFromBeginning) {
        this.skipNumLines = skipNumLines;
        this.delimiter = delimiter;
        this.ratioData = ratioData;
        this.startFromBeginning = startFromBeginning;
    }


    /**
     * This only possible in the directory only contains files of the designated format to read in
     * @param split
     * @throws IOException
     * @throws InterruptedException
     */
    @Override
    public void initialize(InputSplit split) throws IOException, InterruptedException
    {
        this.inputSplit = split;
        List<String> bufferList = new ArrayList<>();

        //this.iter = getIterator(0);

        if (split instanceof FileSplit) {
            this.locations = split.locations();

            if (locations != null && locations.length > 1) {

                for (URI location : locations)
                {

                    boolean toSkipLines = false;
                    int linesToSkip = 0; // new File

                    if(linesToSkip < skipNumLines)
                    {
                        toSkipLines = true;
                    }

                    File file = new File(location);

                    /**
                     * Get file name as label
                     */
                    String[] pathsSplit = file.toString().split("/");
                    String fileName = pathsSplit[pathsSplit.length - 1];
                    String label = fileName.split(Pattern.quote("."))[0];//separate by . , like file.csv to get file name as label

                    System.out.println("Label: " + label);


                    /**
                     * Get data
                     */

                    Iterator<String> iter1 = IOUtils.lineIterator(new InputStreamReader(location.toURL().openStream()));

                    while(iter1.hasNext())
                    {
                        if (toSkipLines && (linesToSkip++ < skipNumLines))
                        {
                            if(linesToSkip >= skipNumLines)
                            {
                                toSkipLines = false;
                            }
                            iter1.next();
                        }
                        else
                        {
                            String data = iter1.next();
                            String newData = data.replace(Character.toString(delimiter), ",").replaceAll("\\[", "").replaceAll("\\]","");

                            bufferList.add(newData);

                            //System.out.println(newData);
                        }

                    }

                }

            }
        }

        Collections.shuffle(bufferList, new Random(super.RANDOM_SEED));


        System.out.println("Total data: " + bufferList.size());
        /**
         * Partition the data
         */
        int partitionStartIndex;
        int partitionEndIndexExclusive;


        if(startFromBeginning == true)
        {
            partitionStartIndex = 0;
            partitionEndIndexExclusive = (int) (Math.ceil(bufferList.size() * ratioData));
        }
        else
        {
            partitionStartIndex = (int) (Math.ceil(bufferList.size() * (1 - ratioData)));
            partitionEndIndexExclusive = bufferList.size();
        }


        dataList = bufferList.subList(partitionStartIndex, partitionEndIndexExclusive);

        this.iter = dataList.iterator();
    }
}
