package org.deeplearning4j.examples.convolution;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;

import java.awt.event.KeyEvent;


/**
 * Created by gtiwari on 1/3/2017.
 */

public class WebCamGrabber implements Runnable
{
        CanvasFrame canvas = new CanvasFrame("Web Cam");

        public WebCamGrabber()
        {
            canvas.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);
        }

        public void run(){

            try
            {
                FrameGrabber grabber = FrameGrabber.createDefault(0); // 1 for next camera
                //mp4 new FFmpegFrameGrabber("video.mp4");

                OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();

                try
                {
                    grabber.start();

                    while (true)
                    {
                        Frame frame = grabber.grab();

                        opencv_core.Mat mt = converter.convert(frame);
                        opencv_core.Mat flipped = new opencv_core.Mat();
                        opencv_core.flip(mt, flipped, 1);

                        canvas.showImage(converter.convert(flipped));

                        KeyEvent t = canvas.waitKey(33);

                        if((t != null) && (t.getKeyCode() == KeyEvent.VK_Q))
                        {
                            break;
                        }


                    }

                    canvas.dispose();
                    grabber.close();
                }
                catch (Exception e)
                {
                    //e.printStackTrace();
                }

            }
            catch(Exception e)
            {

            }

        }

        public static void main(String[] args)
        {
            WebCamGrabber gs = new WebCamGrabber();
            Thread th = new Thread(gs);
            th.start();
        }


    }








