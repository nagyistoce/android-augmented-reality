package com.googlecode.ardemo;

import java.util.ArrayList;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Point3;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

public class CirclesDetector implements FrameProcessor {
    public CirclesDetector() {
        axisPoints.add(new Point3(0, 0, 0));
        axisPoints.add(new Point3(5, 0, 0));
        axisPoints.add(new Point3(0, 5, 0));
        axisPoints.add(new Point3(0, 0, 5));
    }

    @Override
    public Mat process(VideoCapture camera) {
        camera.retrieve(imgRgba, Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
        Imgproc.cvtColor(imgRgba, imgGray, Imgproc.COLOR_RGBA2GRAY);

        points3d.clear();
        for (int idx : points4pnp) {
            Point p = pointNum2XY(idx, 1.0f);
            points3d.add(new Point3(p));
        }

        ptvec.clear();
        boolean found = false;

        if (prevFrameFound) {
            roi.x = Math.max(0, prevWindow.x - prevWindow.width / 2);
            roi.y = Math.max(0, prevWindow.y - prevWindow.height / 2);
            roi.width = Math.min(prevWindow.width * 2, imgGray.cols() - roi.x);
            roi.height = Math.min(prevWindow.height * 2, imgGray.rows() - roi.y);

            found = Calib3d.findCirclesGridDefault(imgGray.submat(roi), patternSize, centers, Calib3d.CALIB_CB_CLUSTERING | Calib3d.CALIB_CB_ASYMMETRIC_GRID);

            if (found) {
                Converters.Mat_to_vector_Point(centers, ptvec);
    
                for (Point p : ptvec) {
                    p.x += roi.x;
                    p.y += roi.y;
                }
            }
        } else {
            found = Calib3d.findCirclesGridDefault(imgGray, patternSize, centers, Calib3d.CALIB_CB_CLUSTERING | Calib3d.CALIB_CB_ASYMMETRIC_GRID);

            if (found)
                Converters.Mat_to_vector_Point(centers, ptvec);
        }

        if (found) {
            prevFrameFound = true;
            prevWindow = Imgproc.boundingRect(ptvec);

            if (drawPoints) {
                for(int k = 0; k < ptvec.size(); ++k) {
                    if (k == points4pnp[0] || k == points4pnp[1] || k == points4pnp[2] || k == points4pnp[3])
                        Core.circle(imgRgba, ptvec.get(k), 3, new Scalar(250, 0, 0));
                    else
                        Core.circle(imgRgba, ptvec.get(k), 3, new Scalar(0, 0, 250));
                }
            }

            // solve pnp
            points2d.clear();
            for (int idx : points4pnp)
                points2d.add(ptvec.get(idx));

            Calib3d.solvePnP(points3d, points2d, getCameraIntrinsics(), getCameraDistortion(), rvec, tvec, false);

            //project axis to image
            Calib3d.projectPoints(Converters.vector_Point3d_to_Mat(axisPoints), rvec, tvec, getCameraIntrinsics(), getCameraDistortion(), imagePointsMat);
            Converters.Mat_to_vector_Point2d(imagePointsMat, imagePoints);

            //draw axis
            Core.line(imgRgba, imagePoints.get(0), imagePoints.get(1), new Scalar(250, 0, 0), 3);
            Core.line(imgRgba, imagePoints.get(0), imagePoints.get(2), new Scalar(0, 250, 0), 3);
            Core.line(imgRgba, imagePoints.get(0), imagePoints.get(3), new Scalar(0, 0, 250), 3);
        } else {
            prevFrameFound = false;
        }

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
    
    private Point pointNum2XY(int idx, float pointsStep) {
        return new Point(pointsStep * 0.5f * (idx / patternWidth), pointsStep * ((idx % patternWidth) + 0.5f * ((idx / patternWidth) % 2)));
    }

    private Mat getNexusCameraIntrinsics() {
        if (intrinsics == null) {
            intrinsics = Mat.eye(3, 3, CvType.CV_32FC1);

            intrinsics.put(0, 0, new float[] {6.2112274782148313e+02f});
            intrinsics.put(0, 1, new float[] {0.0f});
            intrinsics.put(0, 2, new float[] {3.2355657773204325e+02f});

            intrinsics.put(1, 0, new float[] {0.0f});
            intrinsics.put(1, 1, new float[] {6.1911939738925764e+02f});
            intrinsics.put(1, 2, new float[] {2.4530286869951010e+02f});

            intrinsics.put(2, 0, new float[] {0.0f});
            intrinsics.put(2, 1, new float[] {0.0f});
            intrinsics.put(2, 2, new float[] {1.0f});
        }

        return intrinsics;
    }

    private Mat getNexusCameraDistortion() {
        if (distCoeffs == null) {
            distCoeffs = Mat.zeros(5, 1, CvType.CV_32FC1);

            distCoeffs.put(0, 0, new float[] {1.3075382690077034e-01f});
            distCoeffs.put(1, 0, new float[] {-2.4023527434109581e-02f});
            distCoeffs.put(2, 0, new float[] {-1.6705976914291858e-03f});
            distCoeffs.put(3, 0, new float[] {5.2146376856990401e-04f});
            distCoeffs.put(4, 0, new float[] {-1.2997313877043186e+00f});
        }

        return distCoeffs;
    }

    private Mat getCameraIntrinsics() {
        return getNexusCameraIntrinsics();

//        if (intrinsics == null) {
//            intrinsics = Mat.eye(3, 3, CvType.CV_32FC1);
//            
//            intrinsics.put(0, 0, new float[] {400.0f});
//            intrinsics.put(1, 1, new float[] {400.0f});
//            intrinsics.put(0, 2, new float[] {640 / 2});
//            intrinsics.put(1, 2, new float[] {480 / 2});
//        }
//
//        return intrinsics;
    }

    private Mat getCameraDistortion() {
        return getNexusCameraDistortion();

//        if (distCoeffs == null)
//            distCoeffs = Mat.zeros(5, 1, CvType.CV_32FC1);
//
//        return distCoeffs;
    }

    private int patternWidth = 4;
    private int patternHeight = 11;
    private boolean drawPoints = true;

    private int[] points4pnp = new int[] {0, patternWidth - 1, patternWidth * (patternHeight - 1), patternWidth * patternHeight - 1};
    private Size patternSize = new Size(patternWidth, patternHeight);
    private Rect roi = new Rect();

    private ArrayList<Point3> points3d = new ArrayList<Point3>(points4pnp.length);
    private ArrayList<Point> ptvec = new ArrayList<Point>();
    private ArrayList<Point> points2d = new ArrayList<Point>(points4pnp.length);
    private ArrayList<Point3> axisPoints = new ArrayList<Point3>(4);
    private ArrayList<Point> imagePoints = new ArrayList<Point>();

    private Mat rvec = new Mat();
    private Mat tvec = new Mat();

    private boolean prevFrameFound = false;
    private Rect prevWindow = null;
    private Mat centers = new Mat();
    private Mat imagePointsMat = new Mat();

    private Mat imgRgba = new Mat();
    private Mat imgGray = new Mat();

    private Mat intrinsics = null;
    private Mat distCoeffs = null;
}
