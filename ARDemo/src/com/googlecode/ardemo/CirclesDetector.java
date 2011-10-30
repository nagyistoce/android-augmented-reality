package com.googlecode.ardemo;

import org.opencv.core.Mat;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;
import org.opencv.imgproc.Imgproc;

public class CirclesDetector implements FrameProcessor {
    @Override
    public Mat process(VideoCapture camera) {
        camera.retrieve(imgRgba, Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
        Imgproc.cvtColor(imgRgba, imgGray, Imgproc.COLOR_RGBA2GRAY);

        processFrame(imgGray.getNativeObjAddr(), imgRgba.getNativeObjAddr(), patternWidth, patternHeight, drawPoints, drawAxis, 
                     rvec.getNativeObjAddr(), tvec.getNativeObjAddr());

        return imgRgba;
    }

    @Override
    public void resume() {
    }

    @Override
    public void release() {
        imgRgba.release();
        imgGray.release();
    }

    private int patternWidth = 4;
    private int patternHeight = 11;
    private boolean drawPoints = true;
    private boolean drawAxis = true;

    private Mat rvec = new Mat();
    private Mat tvec = new Mat();

    private Mat imgRgba = new Mat();
    private Mat imgGray = new Mat();

    private static native void processFrame(long frame, long outFrame, int patternWidth, int patternHeight, 
                                            boolean drawPoints, boolean drawAxis, long rvec, long tvec);

    static {
        System.loadLibrary("ARDemo");
     }
}
