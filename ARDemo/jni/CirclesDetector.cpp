#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <android/log.h>

#include <jni.h>

#define LOG_TAG "ARDemo"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))
#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__))
#define LOGW(...) ((void)__android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__))
#define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__))

using namespace cv;
using namespace std;

namespace
{
    Mat getNexusCameraIntrinsics()
    {
        static Mat intrinsics;

        if (intrinsics.empty())
        {
            intrinsics = Mat::eye(3, 3, CV_32FC1);
            intrinsics.at<float>(0, 0) = 6.2112274782148313e+02f;
            intrinsics.at<float>(0, 1) = 0.0f;
            intrinsics.at<float>(0, 2) = 3.2355657773204325e+02f;
            intrinsics.at<float>(1, 0) = 0.0f;
            intrinsics.at<float>(1, 1) = 6.1911939738925764e+02f;
            intrinsics.at<float>(1, 2) = 2.4530286869951010e+02f;
            intrinsics.at<float>(2, 0) = 0.0f;
            intrinsics.at<float>(2, 1) = 0.0f;
            intrinsics.at<float>(2, 2) = 1.0f;
        }

        return intrinsics;
    }

    Mat getNexusCameraDistortion()
    {
        static Mat dist_coeffs;

        if (dist_coeffs.empty())
        {
            dist_coeffs = Mat::zeros(5, 1, CV_32FC1);
            dist_coeffs.at<float>(0, 0) = 1.3075382690077034e-01f;
            dist_coeffs.at<float>(1, 0) = -2.4023527434109581e-02f;
            dist_coeffs.at<float>(2, 0) = -1.6705976914291858e-03f;
            dist_coeffs.at<float>(3, 0) = 5.2146376856990401e-04f;
            dist_coeffs.at<float>(4, 0) = -1.2997313877043186e+00f;
        }

        return dist_coeffs;
    }

    Mat getCameraIntrinsics()
    {
        return getNexusCameraIntrinsics();
    }

    Mat getCameraDistortion()
    {
        return getNexusCameraDistortion();
    }

    Point2f pointNum2XY(int idx, float pointsStep, int patternWidth)
    {
        return Point2f(pointsStep * 0.5f * (idx / patternWidth), pointsStep * ((idx % patternWidth) + 0.5f * ((idx / patternWidth) % 2)));
    }

    Ptr<FeatureDetector> getDefaultDetector()
    {
        static Ptr<FeatureDetector> detector;

        if (detector.empty())
        {
            SimpleBlobDetector::Params p;

            p.thresholdStep = 20;
            p.minThreshold = 100;
            p.maxThreshold = 101;
            p.minDistBetweenBlobs = 10;
            p.minRepeatability = 1;
            p.filterByColor = true;
            p.blobColor = 0;

            p.filterByArea = true;
            p.minArea = 5;
            p.maxArea = 5000;

            p.filterByInertia = false;
            p.minInertiaRatio = 0.1f;
            p.maxInertiaRatio = 1.0f;

            p.filterByConvexity = false;
            p.minConvexity = 0.95f;
            p.maxConvexity = 1.0f;

            p.filterByCircularity = false;
            p.minCircularity = 0.6f;
            p.maxCircularity = 1.0f;

            detector = new SimpleBlobDetector(p);
        }

        return detector;
    }

    void processFrame(const Mat& frame, Mat& outFrame, int patternWidth, int patternHeight, bool drawPoints, bool drawAxis, Mat& rvec, Mat& tvec)
    {
        const int points4pnp[] = {0, patternWidth - 1, patternWidth * (patternHeight - 1), patternWidth * patternHeight - 1};
        const int points4pnpNum = sizeof(points4pnp) / sizeof(points4pnp[0]);

        Size patternSize(patternWidth, patternHeight);

        static vector<Point3f> points3d(points4pnpNum);
        for (int i = 0; i < points4pnpNum; ++i)
            points3d[i] = Point3f(pointNum2XY(points4pnp[i], 1.0f, patternWidth));

        static vector<Point2f> ptvec;
        ptvec.clear();

        bool found = false;
        static bool prevFrameFound = false;
        static Rect prevWindow;

        if (prevFrameFound)
        {
            Rect roi;

            roi.x = max(0, prevWindow.x - prevWindow.width / 2);
            roi.y = max(0, prevWindow.y - prevWindow.height / 2);
            roi.width = min(prevWindow.width * 2, frame.cols - roi.x);
            roi.height = min(prevWindow.height * 2, frame.rows - roi.y);

            found = findCirclesGrid(Mat(frame, roi), patternSize, ptvec, CALIB_CB_CLUSTERING | CALIB_CB_ASYMMETRIC_GRID, getDefaultDetector());

            if (found)
            {
                for (size_t i = 0, size = ptvec.size(); i < size; ++i)
                {
                    ptvec[i].x += roi.x;
                    ptvec[i].y += roi.y;
                }
            }
        }
        else
        {
            found = findCirclesGrid(frame, patternSize, ptvec, CALIB_CB_CLUSTERING | CALIB_CB_ASYMMETRIC_GRID, getDefaultDetector());
        }

        if (found)
        {
            prevFrameFound = true;
            prevWindow = boundingRect(ptvec);

            if (drawPoints)
            {
                //draw all found points
                for (size_t i = 0, size = ptvec.size(); i < size; ++i)
                {
                    if (i == points4pnp[0] || i == points4pnp[1] || i == points4pnp[2] || i == points4pnp[3])
                        circle(outFrame, ptvec[i], 3, Scalar(250, 0, 0));
                    else
                        circle(outFrame, ptvec[i], 3, Scalar(0, 0, 250));
                }
            }

            //solve pnp
            static vector<Point2f> points2d(points4pnpNum);
            for (int i = 0; i < points4pnpNum; ++i)
                points2d[i] = Point2f(ptvec[points4pnp[i]]);

            solvePnP(points3d, points2d, getCameraIntrinsics(), getCameraDistortion(), rvec, tvec, false);

            if (drawAxis)
            {
                //project axis to image
                static vector<Point3f> axisPoints;

                if (axisPoints.empty())
                {
                    axisPoints.resize(4);
                    axisPoints[0] = Point3f(0,0,0);
                    axisPoints[1] = Point3f(5,0,0);
                    axisPoints[2] = Point3f(0,5,0);
                    axisPoints[3] = Point3f(0,0,5);
                }

                static vector<Point2f> imagePoints;
                imagePoints.clear();
                projectPoints(axisPoints, rvec, tvec, getCameraIntrinsics(), getCameraDistortion(), imagePoints);

                //draw axis
                line(outFrame, imagePoints[0], imagePoints[1], Scalar(250, 0, 0), 3);
                line(outFrame, imagePoints[0], imagePoints[2], Scalar(0, 250, 0), 3);
                line(outFrame, imagePoints[0], imagePoints[3], Scalar(0, 0, 250), 3);
            }
        }
        else
        {
            prevFrameFound = false;
        }
    }
}

extern "C"
{
    JNIEXPORT void JNICALL Java_com_googlecode_ardemo_CirclesDetector_processFrame(JNIEnv*, jclass,
            jlong frame_, jlong outFrame_, jint patternWidth, jint patternHeight, jboolean drawPoints, jboolean drawAxis, jlong rvec_, jlong tvec_)
    {
        Mat* frame = (Mat*)frame_;
        Mat* outFrame = (Mat*)outFrame_;
        Mat* rvec = (Mat*)rvec_;
        Mat* tvec = (Mat*)tvec_;

        processFrame(*frame, *outFrame, patternWidth, patternHeight, drawPoints, drawAxis, *rvec, *tvec);
    }
}
