package com.googlecode.ardemo;

import org.opencv.core.Mat;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;
import org.opencv.imgproc.Imgproc;

public class QrDetector implements Detector {

    @Override
    public Mat process(VideoCapture camera) {
        camera.retrieve(imgRgba, Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
        Imgproc.cvtColor(imgRgba, imgGray, Imgproc.COLOR_RGBA2GRAY);

        processFrame(imgGray.getNativeObjAddr(), imgRgba.getNativeObjAddr(), drawPoints, drawAxis, 
                     rvec.getNativeObjAddr(), tvec.getNativeObjAddr());

        return imgRgba;
    }

    @Override
    public void resume() {
        releaseNative();
    }

    @Override
    public void release() {
        imgRgba.release();
        imgGray.release();
    }

    private Mat imgRgba = new Mat();
    private Mat imgGray = new Mat();

    private boolean drawPoints = true;
    private boolean drawAxis = true;

    private Mat rvec = new Mat();
    private Mat tvec = new Mat();
    
    private static native void processFrame(long frame, long outFrame, 
            boolean drawPoints, boolean drawAxis, long rvec, long tvec);
    private static native void releaseNative();

    static {
        System.loadLibrary("ARDemo");
    }
}
