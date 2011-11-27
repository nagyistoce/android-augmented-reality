package com.googlecode.ardemo;

import org.opencv.core.Mat;
import org.opencv.highgui.VideoCapture;

public interface Detector {
    Mat process(VideoCapture camera);

    void resume();

    void release();
}
