package org.deeplearning4j.examples.feedforward.classification.wineClassification.old.getWineType;

import org.apache.commons.io.IOUtils;
import org.apache.commons.io.LineIterator;
import org.datavec.api.records.reader.impl.LineRecordReader;
import org.datavec.api.records.reader.impl.csv.SerializableCSVParser;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.split.InputStreamInputSplit;
import org.datavec.api.split.StringSplit;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.*;
import java.util.regex.Pattern;



public class WineTypeRecordReader extends LineRecordReader
{

    protected InputSplit inputSplit;
    protected URI[] locations;

    protected int skipNumLines = 0;
    protected int dataStartIndex = 0;
    protected int dataEndIndex = 0;

    protected Iterator<String> iter; //for iteration in next

    private SerializableCSVParser csvParser;
    public static char delimiter = ',';
    private static boolean iterEnd = false;
    public static List<String> labels;

    protected List<String> dataList = new ArrayList<>();
    protected static final int RANDOM_SEED = 123;

    public WineTypeRecordReader()
    {

    }

    public WineTypeRecordReader(int skipNumLines, char delimiter, int dataStartIndex, int dataEndIndexExclusive)
    {
        this.skipNumLines = skipNumLines;
        this.delimiter = delimiter;
        this.dataStartIndex = dataStartIndex + skipNumLines;
        this.dataEndIndex = dataEndIndexExclusive + skipNumLines;
    }

    @Override
    public boolean hasNext() {

        if(iterEnd == false && iter == null)
        {
            iter = dataList.iterator();
            return true;
        }
        else if(iter.hasNext())
        {
            return true;
        }

        return false;

    }

    @Override
    public List<Writable> next()
    {
        if (iter.hasNext())
        {
            /**
             * Important block in here
             */
            List<Writable> ret = new ArrayList<>();
            String currentRecord = iter.next();
            String[] temp = currentRecord.split(",");

            for(int i = 0;i < temp.length; i++)
            {
                ret.add(new DoubleWritable(Double.parseDouble(temp[i])));
            }
            return ret;
        }
        else
            throw new IllegalStateException("no more elements");

    }

    @Override
    public void reset() {
        iterEnd = false;
        iter = dataList.iterator();
    }

    /*
    protected Iterator<String> getIterator(int location) {
        Iterator<String> iterator = null;
        if (inputSplit instanceof StringSplit) {
            StringSplit stringSplit = (StringSplit) inputSplit;
            iterator = Collections.singletonList(stringSplit.getData()).listIterator();
        } else if (inputSplit instanceof InputStreamInputSplit) {
            InputStream is = ((InputStreamInputSplit) inputSplit).getIs();
            if (is != null) {
                iterator = IOUtils.lineIterator(new InputStreamReader(is));
            }
        } else {
            this.locations = inputSplit.locations();
            if (locations != null && locations.length > 0) {
                InputStream inputStream;
                try {
                    inputStream = locations[location].toURL().openStream();
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
                iterator = IOUtils.lineIterator(new InputStreamReader(inputStream));
            }
        }
        if (iterator == null)
            throw new UnsupportedOperationException("Unknown input split: " + inputSplit);
        return iterator;
    }
    */

    protected void closeIfRequired(Iterator<String> iterator) {
        if (iterator instanceof LineIterator) {
            LineIterator iter = (LineIterator) iterator;
            iter.close();
        }
    }

    protected List<Writable> parseLine(String line) {
        String[] split;
        try {
            split = csvParser.parseLine(line);
        } catch(IOException e) {
            throw new RuntimeException(e);
        }
        List<Writable> ret = new ArrayList<>();
        for (String s : split) {
            ret.add(new Text(s));
        }
        return ret;
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
        int iterLabel = 0;

        this.inputSplit = split;
        //this.iter = getIterator(0);

        if (split instanceof FileSplit) {
            this.locations = split.locations();

            if (locations != null && locations.length > 1) {

                for (URI location : locations)
                {

                    boolean toSkipLines = false;
                    int linesToSkip = 0; // new File
                    int dataIndex = 0;

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
                            ++dataIndex;
                        }
                        else if(dataIndex < dataStartIndex)
                        {
                            iter1.next();
                            ++dataIndex;
                        }
                        else if(dataIndex >= dataEndIndex)
                        {
                            break;
                        }
                        else
                        {
                            String data = iter1.next() + "," + String.valueOf(iterLabel);
                            String newData = data.replace(Character.toString(delimiter), ",").replaceAll("\\[", "").replaceAll("\\]","");

                            dataList.add(newData);

                            //System.out.println(newData);
                            ++dataIndex;
                        }

                    }

                     ++iterLabel;

                }

            }
        }

        Collections.shuffle(dataList, new Random(RANDOM_SEED));
        this.iter = dataList.iterator();
    }

}
