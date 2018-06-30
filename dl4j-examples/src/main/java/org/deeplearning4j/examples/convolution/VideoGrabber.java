package org.deeplearning4j.examples.convolution;

import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;


/**
 * Created by gtiwari on 1/3/2017.
 */

public class VideoGrabber implements Runnable
{
    final int INTERVAL = 100;///you may use interval
    CanvasFrame canvas = new CanvasFrame("Web Cam");

    public VideoGrabber()
    {
        canvas.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);
    }

    public void run(){

        try
        {
            //FmpegFrameGrabber grabber = new FFmpegFrameGrabber("/Users/wei/Desktop/Skymind/Videos/videoSample.mp4");


            FrameGrabber grabber = FrameGrabber.createDefault("/Users/wei/Desktop/Skymind/Videos/videoSample.mp4");

            try {
                grabber.start();

                while (true)
                {
                    try
                    {
                        Frame frame = grabber.grab();
                        canvas.showImage(frame);
                    }
                    catch(FrameGrabber.Exception e)
                    {
                        break;
                    }

                }

                canvas.dispose();

            } catch (Exception e) {
                //e.printStackTrace();
            }

        }catch(Exception e)
        {

        }

    }

    public static void main(String[] args) {
        VideoGrabber gs = new VideoGrabber();
        Thread th = new Thread(gs);
        th.start();
    }


}








