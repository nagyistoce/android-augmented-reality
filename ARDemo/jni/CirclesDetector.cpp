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
    const int c_points4pnpNum = 4;

    template <typename T> inline void releaseVector(vector<T>& v)
    {
        vector<T> empty;
        empty.swap(v);
    }

    class Buffers
    {
    public:
        static Mat intrinsics;
        static Mat dist_coeffs;
        static Ptr<FeatureDetector> detector;
        static vector<Point3f> points3d;
        static vector<Point2f> ptvec;
        static vector<Point2f> points2d;
        static vector<Point3f> axisPoints;
        static vector<Point2f> imagePoints;

        static void resume()
        {
            createNexusCameraIntrinsics();
            createNexusCameraDistortion();
            createSimpleBlobDetector();
            points3d.resize(c_points4pnpNum);
            points2d.resize(c_points4pnpNum);
            axisPoints.resize(4);
            axisPoints[0] = Point3f(0,0,0);
            axisPoints[1] = Point3f(5,0,0);
            axisPoints[2] = Point3f(0,5,0);
            axisPoints[3] = Point3f(0,0,5);
        }

        static void release()
        {
            intrinsics.release();
            dist_coeffs.release();
            detector.release();
            releaseVector(points3d);
            releaseVector(ptvec);
            releaseVector(points2d);
            releaseVector(axisPoints);
            releaseVector(imagePoints);
        }

    private:
        static void createNexusCameraIntrinsics()
        {
            intrinsics.create(3, 3, CV_32FC1);

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

        static void createNexusCameraDistortion()
        {
            dist_coeffs.create(5, 1, CV_32FC1);

            dist_coeffs.at<float>(0, 0) = 1.3075382690077034e-01f;
            dist_coeffs.at<float>(1, 0) = -2.4023527434109581e-02f;
            dist_coeffs.at<float>(2, 0) = -1.6705976914291858e-03f;
            dist_coeffs.at<float>(3, 0) = 5.2146376856990401e-04f;
            dist_coeffs.at<float>(4, 0) = -1.2997313877043186e+00f;
        }

        static void createSimpleBlobDetector()
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
    };

    Mat Buffers::intrinsics;
    Mat Buffers::dist_coeffs;
    Ptr<FeatureDetector> Buffers::detector;
    vector<Point3f> Buffers::points3d;
    vector<Point2f> Buffers::ptvec;
    vector<Point2f> Buffers::points2d;
    vector<Point3f> Buffers::axisPoints;
    vector<Point2f> Buffers::imagePoints;

    inline Point2f pointNum2XY(int idx, float pointsStep, int patternWidth)
    {
        return Point2f(pointsStep * 0.5f * (idx / patternWidth), pointsStep * ((idx % patternWidth) + 0.5f * ((idx / patternWidth) % 2)));
    }

    void processFrame(const Mat& frame, Mat& outFrame, int patternWidth, int patternHeight, bool drawPoints, bool drawAxis, Mat& rvec, Mat& tvec)
    {
        const int points4pnp[] = {0, patternWidth - 1, patternWidth * (patternHeight - 1), patternWidth * patternHeight - 1};

        Size patternSize(patternWidth, patternHeight);

        for (int i = 0; i < c_points4pnpNum; ++i)
        {
            Point2f p = pointNum2XY(points4pnp[i], 1.0f, patternWidth);
            Buffers::points3d[i] = Point3f(p.y, 0.0f, p.x);
        }

        Buffers::ptvec.clear();

        bool found = false;
        static bool prevFrameFound = false;
        static Rect prevWindow;

        if (!prevFrameFound)
            found = findCirclesGrid(frame, patternSize, Buffers::ptvec, CALIB_CB_CLUSTERING | CALIB_CB_ASYMMETRIC_GRID, Buffers::detector);
        else
        {
            Rect roi;

            roi.x = max(0, prevWindow.x - prevWindow.width / 2);
            roi.y = max(0, prevWindow.y - prevWindow.height / 2);
            roi.width = min(prevWindow.width * 2, frame.cols - roi.x);
            roi.height = min(prevWindow.height * 2, frame.rows - roi.y);

            found = findCirclesGrid(Mat(frame, roi), patternSize, Buffers::ptvec, CALIB_CB_CLUSTERING | CALIB_CB_ASYMMETRIC_GRID, Buffers::detector);

            if (found)
            {
                for (size_t i = 0, size = Buffers::ptvec.size(); i < size; ++i)
                {
                    Buffers::ptvec[i].x += roi.x;
                    Buffers::ptvec[i].y += roi.y;
                }
            }
        }

        if (!found)
            prevFrameFound = false;
        else
        {
            prevFrameFound = true;
            prevWindow = boundingRect(Buffers::ptvec);

            if (drawPoints)
            {
                //draw all found points
                for (size_t i = 0, size = Buffers::ptvec.size(); i < size; ++i)
                {
                    if (i == points4pnp[0] || i == points4pnp[1] || i == points4pnp[2] || i == points4pnp[3])
                        circle(outFrame, Buffers::ptvec[i], 3, Scalar(250, 0, 0));
                    else
                        circle(outFrame, Buffers::ptvec[i], 3, Scalar(0, 0, 250));
                }
            }

            //solve pnp
            for (int i = 0; i < c_points4pnpNum; ++i)
                Buffers::points2d[i] = Buffers::ptvec[points4pnp[i]];

            solvePnP(Buffers::points3d, Buffers::points2d, Buffers::intrinsics, Buffers::dist_coeffs, rvec, tvec, false);

            if (drawAxis)
            {
                //project axis to image
                Buffers::imagePoints.clear();
                projectPoints(Buffers::axisPoints, rvec, tvec, Buffers::intrinsics, Buffers::dist_coeffs, Buffers::imagePoints);

                //draw axis
                line(outFrame, Buffers::imagePoints[0], Buffers::imagePoints[1], Scalar(250, 0, 0), 3);
                line(outFrame, Buffers::imagePoints[0], Buffers::imagePoints[2], Scalar(0, 250, 0), 3);
                line(outFrame, Buffers::imagePoints[0], Buffers::imagePoints[3], Scalar(0, 0, 250), 3);
            }
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

    JNIEXPORT void JNICALL Java_com_googlecode_ardemo_CirclesDetector_resumeNative(JNIEnv*, jclass)
    {
        Buffers::resume();
    }

    JNIEXPORT void JNICALL Java_com_googlecode_ardemo_CirclesDetector_releaseNative(JNIEnv*, jclass)
    {
        Buffers::release();
    }
}
