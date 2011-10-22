package com.googlecode.ardemo;

import java.text.DecimalFormat;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;

public class FpsMeter {
    public void measure() {
        ++frames;

        if (frames == STEP) {
            frames = 0;
            long time = Core.getTickCount();
            fps = STEP * FREQ / (time - prevTime);
            prevTime = time;
        }
    }

    public void draw(Mat img) {
        String fpsstr = "FPS: " + FPS_FMT.format(fps);

        double fontScale = FONT_SCALE_480 * img.rows() / 480.0;
        Size size = Core.getTextSize(fpsstr, FONT_FACE, fontScale, 1, null);
        rectRightBottom.x = RECT_LEFT_TOP.x + size.width;
        rectRightBottom.y = RECT_LEFT_TOP.y + size.height;
        textOrigin.x = RECT_LEFT_TOP.x;
        textOrigin.y = rectRightBottom.y;

        Core.rectangle(img, RECT_LEFT_TOP, rectRightBottom, RECT_COLOR, Core.FILLED);
        Core.putText(img, fpsstr, textOrigin, FONT_FACE, fontScale, TEXT_COLOR);
    }

    private int    frames          = 0;
    private long   prevTime        = 0;
    private double fps             = 0.0;

    private Point  rectRightBottom = new Point();
    private Point  textOrigin      = new Point();

    private static final int STEP = 20;

    private static final double FREQ = Core.getTickFrequency();

    private static final DecimalFormat FPS_FMT = new DecimalFormat("0.0");

    private static final int    FONT_FACE      = Core.FONT_HERSHEY_TRIPLEX;
    private static final double FONT_SCALE_480 = 1.3;

    private static final Point  RECT_LEFT_TOP = new Point(1.0, 1.0);
    private static final Scalar RECT_COLOR    = new Scalar(0.0, 0.0, 0.0, 255.0);

    private static final Scalar TEXT_COLOR = new Scalar(255.0, 255.0, 255.0, 255.0);
}
