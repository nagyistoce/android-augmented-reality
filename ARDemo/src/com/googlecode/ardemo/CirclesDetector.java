package com.googlecode.ardemo;

import org.opencv.core.Mat;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;
import org.opencv.imgproc.Imgproc;

public class CirclesDetector implements Detector {
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
        resumeNative();
    }

    @Override
    public void release() {
        imgRgba.release();
        imgGray.release();
        releaseNative();
    }

    private final int patternWidth = 4;
    private final int patternHeight = 11;
    private final boolean drawPoints = true;
    private final boolean drawAxis = true;

    private final Mat rvec = new Mat();
    private final Mat tvec = new Mat();

    private final Mat imgRgba = new Mat();
    private final Mat imgGray = new Mat();

    private static native void processFrame(long frame, long outFrame, int patternWidth, int patternHeight, 
                                            boolean drawPoints, boolean drawAxis, long rvec, long tvec);
    private static native void resumeNative();
    private static native void releaseNative();

    static {
        System.loadLibrary("ARDemo");
    }
}
