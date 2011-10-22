package com.googlecode.ardemo;

import java.util.List;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;

import android.graphics.Bitmap;
import android.util.Log;

public class CameraHolder {
    public void toggleFps() {
        showFps = !showFps;
    }

    public List<Size> getAvaibleCameraResolutions() {
        return cameraResolutions;
    }

    public synchronized void setCameraResolution(Size resolution) {
        this.resolution = resolution;

        if (camera != null) {
            Log.i(TAG, "Set camera resolution: " + resolution);

            camera.set(Highgui.CV_CAP_PROP_FRAME_WIDTH, resolution.width);
            camera.set(Highgui.CV_CAP_PROP_FRAME_HEIGHT, resolution.height);
        }
    }

    public synchronized void setFrameProcessor(FrameProcessor frameProc) {
        if (this.frameProc != null)
            this.frameProc.release();

        this.frameProc = frameProc;
        this.frameProc.resume();
    }

    public synchronized void create() {
        Log.i(TAG, "Try to create native camera");

        // Release, if we have previous resources
        releaseImpl();

        camera = new VideoCapture(Highgui.CV_CAP_ANDROID);

        if (camera.isOpened()) {
            Log.i(TAG, "Native camera was created");

            cameraResolutions = camera.getSupportedPreviewSizes();
        }
        else {
            Log.e(TAG, "Can't create native camera");

            camera.release();
            camera = null;
        }
    }

    public synchronized void updateResolution(int width, int height) {
        if (camera != null && camera.isOpened()) {
            if (resolution.height == 0 || resolution.width == 0) {
                Log.i(TAG, "Choose best camera resolution");

                double minDiff = Double.MAX_VALUE;
                for (Size size : cameraResolutions) {
                    double diff = Math.abs(size.height - height);
                    if (diff < minDiff) {
                        resolution.width = size.width;
                        resolution.height = size.height;
                        minDiff = diff;
                    }
                }

                Log.i(TAG, "Set camera resolution: " + resolution);

                camera.set(Highgui.CV_CAP_PROP_FRAME_WIDTH, resolution.width);
                camera.set(Highgui.CV_CAP_PROP_FRAME_HEIGHT, resolution.height);
            }
        }
    }

    public synchronized Bitmap processFrame() {
        Log.i(TAG, "Process frame");

        fps.measure();

        if (camera == null)
            return null;

        if (!camera.grab()) {
            Log.e(TAG, "Can't retriev frame from camera");
            return null;
        }

        Mat img = null;
        if (frameProc != null) {
            Log.i(TAG, "Process frame");

            img = frameProc.process(camera);
        }

        if (img != null && img.type() == CvType.CV_8UC4) {
            if (showFps)
                fps.draw(img);

            if (bmp == null || bmp.getWidth() != img.cols() || bmp.getHeight() != img.rows()) {
                Log.i(TAG, "Create bitmap object");

                if (bmp != null)
                    bmp.recycle();

                bmp = Bitmap.createBitmap(img.cols(), img.rows(), Bitmap.Config.ARGB_8888);
            }

            Log.i(TAG, "Convert Mat object to Bitmap");

            if (Utils.matToBitmap(img, bmp)) {
                Log.i(TAG, "Mat was converted");

                return bmp;
            }
        }

        return null;
    }

    public synchronized void resume() {
        frameProc.resume();
    }

    public synchronized void release() {
        releaseImpl();
    }

    private void releaseImpl() {
        if (camera != null) {
            Log.i(TAG, "Release native camera");

            camera.release();
        }
        camera = null;

        cameraResolutions = null;
        resolution = new Size();

        if (bmp != null) {
            Log.i(TAG, "Release Bitmap object");

            bmp.recycle();
        }
        bmp = null;

        if (frameProc != null)
            frameProc.release();
    }

    private FrameProcessor frameProc = null;

    private VideoCapture camera            = null;
    private List<Size>   cameraResolutions = null;
    private Size         resolution        = new Size();

    private FpsMeter fps     = new FpsMeter();
    private boolean  showFps = false;

    private Bitmap bmp = null;

    private static final String TAG = "UltraEye::CameraHolder";
}
