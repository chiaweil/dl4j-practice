package org.deeplearning4j.examples.convolution;

import org.bytedeco.javacv.CanvasFrame;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.videoio.VideoCapture;

import static org.opencv.highgui.HighGui.imshow;


/**
 * Created by gtiwari on 1/3/2017.
 */

public class WebCamOld implements Runnable
{
    final int INTERVAL = 100;///you may use interval
    CanvasFrame canvas = new CanvasFrame("Web Cam");

    public WebCamOld()
    {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        canvas.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);
    }

    public void run(){

        VideoCapture vc = new VideoCapture(0);

        Mat mat = new Mat();

        while(true)
        {

            if(vc.read(mat))
            {
                imshow("mat", mat);

            }


        }

    }

    public static void main(String[] args) {
        WebCamOld gs = new WebCamOld();
        Thread th = new Thread(gs);
        th.start();
    }


}


/*
try
        {
            FrameGrabber grabber =  new OpenCVFrameConverter(0);//FrameGrabber.createDefault(0);//VideoInputFrameGrabber(0); // 1 for next camera
            OpenCVFrameConverter.ToIplImage converter = new OpenCVFrameConverter.ToIplImage();
            IplImage img;
            int i = 0;

            try {
                grabber.start();
                while (true) {
                    Frame frame = grabber.grab();

                    img = converter.convert(frame);

                    //the grabbed frame will be flipped, re-flip to make it right
                    cvFlip(img, img, 1);// l-r = 90_degrees_steps_anti_clockwise

                    canvas.showImage(converter.convert(img));

                    Thread.sleep(INTERVAL);
                }
            } catch (Exception e) {
                //e.printStackTrace();
            }

        }
        catch(FrameGrabber.Exception e)
        {
            System.out.println("FrameGrabber Exception caught");
        }




 */
