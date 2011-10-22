package com.googlecode.ardemo;

import org.opencv.core.Mat;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;

public class EmptyProcessor implements FrameProcessor {

    @Override
    public Mat process(VideoCapture camera) {
        camera.retrieve(img, Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
        return img;
    }

    @Override
    public void resume() {
    }

    @Override
    public void release() {
        img.release();
    }

    private Mat img = new Mat();
}
