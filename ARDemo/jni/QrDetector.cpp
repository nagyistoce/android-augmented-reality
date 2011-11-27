// The code is based on libdecodeqr sources:
// Copyright(C) 2007 NISHI Takao <zophos@koka-in.org>
//                   JMA  (Japan Medical Association)
//                   NaCl (Network Applied Communication Laboratory Ltd.)

#include <vector>
#include <algorithm>

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

namespace Qr
{
    const int DEFAULT_ADAPTIVE_TH_SIZE = 25;
    const int DEFAULT_ADAPTIVE_TH_DELTA = 10;

    const double DEFAULT_MIN_AREA = 49;
    const double DEFAULT_MIN_AREA_RATIO = 0.65;
    const double DEFAULT_MIN_FERET_RATIO = 0.7;

    const double DEFAULT_FIND_CODE_AREA_POLY_APPROX_TH = 3;
    const double DEFAULT_BOX_SIZE_FACTOR = 4.0f / 7.0f;

    const int DEFAULT_POSTERIZED_TH_LOW = 64;
    const int DEFAULT_POSTERIZED_TH_HI = 96;
    const int DEFAULT_POSTERIZED_TH_STEP = 8;

    void binarize(const cv::Mat& src, cv::Mat& binarized,
        int adaptive_th_size = DEFAULT_ADAPTIVE_TH_SIZE, int adaptive_th_delta = DEFAULT_ADAPTIVE_TH_DELTA);

    void findBorderPattern(cv::Mat& image, std::vector<cv::RotatedRect>& borderPattern,
        double minArea = DEFAULT_MIN_AREA, double minAreaRatio = DEFAULT_MIN_AREA_RATIO, double minFeretRation = DEFAULT_MIN_FERET_RATIO);

    void findCodeAreaContour(cv::Mat& image, std::vector<cv::RotatedRect>& borderPattern, std::vector<cv::Point>& codeAreaContour,
        double th = DEFAULT_FIND_CODE_AREA_POLY_APPROX_TH, double boxSizeFactor = DEFAULT_BOX_SIZE_FACTOR);

    void transformImage(const cv::Mat& image, cv::Mat& transformed,
        std::vector<cv::Point>& codeAreaContour, const std::vector<cv::RotatedRect>& boxesPattern);

    void posterizeImage(const cv::Mat& image, const std::vector<cv::RotatedRect>& boxesPattern, cv::Mat& code,
                    int th_low = DEFAULT_POSTERIZED_TH_LOW, int th_hi = DEFAULT_POSTERIZED_TH_HI, int th_step = DEFAULT_POSTERIZED_TH_STEP,
                    int adaptive_th_size = DEFAULT_ADAPTIVE_TH_SIZE, int adaptive_th_delta = DEFAULT_ADAPTIVE_TH_DELTA);
}

////////////////////////////////////////////////////////////////////////////////////////
// binarize

namespace
{
    Mat binarizeBuf;
}

void Qr::binarize(const Mat& src, Mat& binarized, int adaptive_th_size, int adaptive_th_delta)
{
    adaptiveThreshold(src, binarized, 128, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, adaptive_th_size, adaptive_th_delta);
//    CV_Assert(src.type() == CV_8UC1);
//
//    medianBlur(src, binarizeBuf, 3);
//
//    if (adaptive_th_size > 0)
//        adaptiveThreshold(binarizeBuf, binarized, 128, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, adaptive_th_size, adaptive_th_delta);
//    else
//        threshold(binarizeBuf, binarized, adaptive_th_delta, 255, THRESH_BINARY_INV);
}

////////////////////////////////////////////////////////////////////////////////////////
// findBorderPattern

namespace
{
    Rect maxRect(const Rect& rect1, const Rect& rect2)
    {
        Rect max_rect;
        int a, b;

        max_rect.x = a = rect1.x;
        b = rect2.x;
        if (max_rect.x > b)
            max_rect.x = b;

        max_rect.width = a += rect1.width;
        b += rect2.width;

        if (max_rect.width < b)
            max_rect.width = b;
        max_rect.width -= max_rect.x;

        max_rect.y = a = rect1.y;
        b = rect2.y;
        if (max_rect.y > b)
            max_rect.y = b;

        max_rect.height = a += rect1.height;
        b += rect2.height;

        if (max_rect.height < b)
            max_rect.height = b;
        max_rect.height -= max_rect.y;

        return max_rect;
    }

    vector< vector<Point> > findBorderPattern_contours;
    vector< pair< Rect, const vector<Point>* > > findBorderPattern_candidates;
}

void Qr::findBorderPattern(Mat& image, vector<RotatedRect>& borderPattern, double minArea, double minAreaRatio, double minFeretRation)
{
    CV_Assert(image.type() == CV_8UC1);

    borderPattern.clear();
    borderPattern.reserve(3);

    //
    // Find all contours.
    //
    findBorderPattern_contours.clear();
    findContours(image, findBorderPattern_contours, RETR_LIST, CHAIN_APPROX_NONE);

    //
    // check each block
    //
    findBorderPattern_candidates.clear();
    findBorderPattern_candidates.reserve(findBorderPattern_contours.size());
    for (size_t i = 0, size = findBorderPattern_contours.size(); i < size; ++i)
    {
        const vector<Point>& contour = findBorderPattern_contours[i];
        Rect feret = boundingRect(contour);

        double area = fabs(contourArea(contour));
        double areaRatio = area / static_cast<double>(feret.width * feret.height);
        double feretRatio = ((feret.width < feret.height)? static_cast<double>(feret.width) / feret.height : static_cast<double>(feret.height) / feret.width);

        //
        // search square
        //
        if (area >= minArea && areaRatio >= minAreaRatio && feretRatio >= minFeretRation)
            findBorderPattern_candidates.push_back(make_pair(feret, &contour));
    }

    //
    // check each sqare has inner squire
    //
    for (size_t i = 0, size = findBorderPattern_candidates.size(); i < size; ++i)
    {
        int innerContour = 0;

        for (size_t j = 0; j < size; ++j)
        {
            if (i == j) continue;

            if (findBorderPattern_candidates[i].first == maxRect(findBorderPattern_candidates[i].first, findBorderPattern_candidates[j].first))
                innerContour++;
        }

        //
        // There were 2 squires (white and black) inside a squire,
        // the most outer squire assumed as position marker.
        //
        if (innerContour == 2)
        {
            RotatedRect box = minAreaRect(*findBorderPattern_candidates[i].second);
            borderPattern.push_back(box);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////
// findCodeAreaContour

void Qr::findCodeAreaContour(Mat& image, vector<RotatedRect>& borderPattern, vector<Point>& codeAreaContour, double th, double boxSizeFactor)
{
    CV_Assert(image.type() == CV_8UC1);
    CV_Assert(borderPattern.size() == 3);

    Mat mask(image.size(), CV_8UC1, Scalar::all(0));

    //
    // create position maker mask
    //
    vector<Point2f> markersVertex;
    markersVertex.reserve(4 * borderPattern.size());
    for (size_t i = 0, size = borderPattern.size(); i < size; ++i)
    {
        RotatedRect& box = borderPattern[i];

        //
        // set 2-cells margin for fale-safe
        //
        box.size.width += static_cast<float>(box.size.width * boxSizeFactor);
        box.size.height += static_cast<float>(box.size.height * boxSizeFactor);

        //
        // get each position maker's vertex
        //
        Point2f pts[4];
        box.points(pts);
        for(int j = 0; j < 4; ++j)
            markersVertex.push_back(pts[j]);
    }

    //
    // create Minimal-area bounding rectangle which condist
    // every position makers
    //
    RotatedRect box = minAreaRect(markersVertex);

    //
    // create code area mask
    //
    Point2f pt_32f[4];
    box.points(pt_32f);

    vector< vector<Point> > points(1);
    points[0].resize(4);
    for (int i = 0; i < 4; ++i)
        points[0][i] = pt_32f[i];

    fillPoly(mask, points, Scalar::all(255));

    //
    //  apply mask to src image and reduce noise using opening
    //
    bitwise_and(image, mask, mask);

    Mat temp;

    erode(mask, temp, Mat());
    swap(mask, temp);

    dilate(mask, temp, Mat());
    swap(mask, temp);

    //
    // get contours of masked image
    //
    vector< vector<Point> > cont;
    findContours(mask, cont, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    //
    // calcurate convex hull that assumed as code area
    //
    vector<Point> pts;
    for (size_t i = 0, size = cont.size(); i < size; ++i)
        pts.insert(pts.end(), cont[i].begin(), cont[i].end());

    vector<Point> hull;
    convexHull(pts, hull);

    //
    // polygonal approximation to reduce noise
    //
    codeAreaContour.clear();
    approxPolyDP(hull, codeAreaContour, th, false);

    hull.swap(codeAreaContour);
    codeAreaContour.clear();

    vector<size_t> indexies;
    indexies.reserve(codeAreaContour.size());

    for (size_t i = 1, size = hull.size(); i < size; ++i)
    {
        size_t left = i;
        size_t rigth = i > 1 ? i - 2 : size - 1;
        size_t middle = i - 1;

        double x1 = hull[left].x;
        double y1 = hull[left].y;
        double x2 = hull[rigth].x;
        double y2 = hull[rigth].y;
        double x0 = hull[middle].x;
        double y0 = hull[middle].y;

        double A = (1.0 - y1 / y2) / (x1 - x2 * y1 / y2);
        double B = -(A * x2 + 1.0) / y2;

        double d = fabs((A * x0 + B * y0 + 1.0) / sqrt(A * A + B * B));

        if (d < th)
            indexies.push_back(middle);
    }

    CV_Assert(indexies.size() < hull.size());

    codeAreaContour.reserve(hull.size() - indexies.size());

    for (size_t i = 0, size = hull.size(); i < size; ++i)
    {
        if (find(indexies.begin(), indexies.end(), i) == indexies.end())
            codeAreaContour.push_back(hull[i]);
    }
}

////////////////////////////////////////////////////////////////////////////////////////
// transformImage

namespace
{
    class ClockWisePred : binary_function<Point, Point, bool>
    {
    public:
        ClockWisePred(const Point2f& cog) : cog_(cog) {}

        bool operator ()(const Point& a, const Point& b) const
        {
            float aa = fastAtan2(a.y - cog_.y, cog_.x - a.x);
            float ba = fastAtan2(b.y - cog_.y, cog_.x - b.x);

            return aa < ba;
        }

    private:
        Point2f cog_;
    };
}

void Qr::transformImage(const Mat& image, Mat& transformed, vector<Point>& codeAreaContour, const vector<RotatedRect>& boxesPattern)
{
    CV_Assert(image.type() == CV_8UC1);
    CV_Assert(codeAreaContour.size() == 4);
    CV_Assert(boxesPattern.size() == 3);

    //
    // get code area's Centor of Gravity
    //
    Point2f cog;
    for (size_t i = 0, size = codeAreaContour.size(); i < size; ++i)
    {
        cog.x += codeAreaContour[i].x;
        cog.y += codeAreaContour[i].y;
    }
    cog.x /= codeAreaContour.size();
    cog.y /= codeAreaContour.size();

    //
    // sort code_area_contour by clock-wise
    //
    sort(codeAreaContour.begin(), codeAreaContour.end(), ClockWisePred(cog));

    //
    // calculates matrix of perspective transform
    //
    // The src rectangle transform to a square which left-top vertex
    // is same as src rectangle and each side length is same as top side
    // length of src.
    //
    Point codeRegionVertexes[4];
    Point2f spts[4];
    Point2f dpts[4];
    float max_d = 0;
    int offset = 0;

    for (size_t i = 0; i < 4; ++i)
    {
        codeRegionVertexes[i] = codeAreaContour[i];

        spts[i] = codeRegionVertexes[i];

        //
        // find nearest finder pattern
        //
        float tmp_d = 0;
        float min_d = (boxesPattern[0].center.x - spts[i].x) * (boxesPattern[0].center.x - spts[i].x)
                    + (boxesPattern[0].center.y - spts[i].y) * (boxesPattern[0].center.y - spts[i].y);

        for (size_t j = 1; j < 3; ++j)
        {
            tmp_d = (boxesPattern[j].center.x - spts[i].x) * (boxesPattern[j].center.x - spts[i].x)
                  + (boxesPattern[j].center.y - spts[i].y) * (boxesPattern[j].center.y - spts[i].y);

            if (min_d > tmp_d)
                min_d = tmp_d;
        }

        if (max_d < min_d)
        {
            max_d = min_d;
            offset = i;
        }
    }

    offset = (offset + 2) % 4;

    float sideLen = sqrt((spts[0].x - spts[1].x) * (spts[0].x - spts[1].x) + (spts[0].y - spts[1].y) * (spts[0].y - spts[1].y));
    for (size_t i = 0; i < 4; ++i)
        dpts[i] = Point2f(0, 0);

    dpts[(offset + 1) % 4].x += sideLen;
    dpts[(offset + 2) % 4].x += sideLen;
    dpts[(offset + 2) % 4].y += sideLen;
    dpts[(offset + 3) % 4].y += sideLen;

    Mat map = getPerspectiveTransform(spts, dpts);

    //
    // perspective transform
    //
    warpPerspective(image, transformed, map, image.size());

    //
    // set ROI as code area
    //
    Rect roi(static_cast<int>(dpts[offset].x), static_cast<int>(dpts[offset].y), static_cast<int>(sideLen + 1.0f), static_cast<int>(sideLen + 1.0f));
    transformed = transformed(roi);
}

////////////////////////////////////////////////////////////////////////////////////////
// posterizeImage

namespace
{
    void apaptiveWhiteLeveling(const Mat& src, Mat& dst, double middle_value, int block_size, double param1)
    {
        Mat mean;
        boxFilter(src, mean, mean.depth(), Size(block_size, block_size));

        subtract(mean, param1, mean);

        absdiff(src, mean, dst);

        Mat mask;
        compare(src, mean, mask, CMP_GT);

        add(dst, middle_value, dst, mask);

        bitwise_not(mask, mask);

        subtract(middle_value, dst, dst, mask);

        bitwise_not(dst, dst);
    }

    void createPosterizedImage(const Mat& src, Mat& dst, int block_size, double delta, int low_th, int hi_th)
    {
        if (block_size > 0)
            apaptiveWhiteLeveling(src, dst, 128, block_size, delta);
        else
            threshold(src, dst, delta, 255, THRESH_BINARY_INV);

        int a = 0;
        int b = 0;
        if (hi_th > low_th)
        {
            a = 128 / (hi_th - low_th);
            b = -a * low_th;
        }
        else
        {
            hi_th = low_th;
        }

        uchar lut_data[256];
        for (int i = 0; i < low_th; ++i)
            lut_data[i] = 0;

        for (int i = low_th; i < hi_th; ++i)
            lut_data[i] = a * i + b;

        for (int i = hi_th; i < 256; ++i)
            lut_data[i] = 255;

        Mat lut(1, 256, CV_8UC1, lut_data);
        Mat buf;
        LUT(dst, lut, buf);
        dst = buf.clone();

        Mat temp;
        dilate(buf, temp, Mat());
        swap(temp, buf);
        erode(buf, temp, Mat());
        swap(temp, buf);

        Rect roiMask(Point(), src.size());
        roiMask.x += 1;
        roiMask.y += 1;
        roiMask.width -= 2;
        roiMask.height -= 2;
        buf(roiMask).setTo(Scalar::all(0));
        bitwise_or(dst, buf, dst);
    }

    double getCellSize(const vector<RotatedRect>& boxesPattern)
    {
        size_t c = boxesPattern.size();
        if (c != 3)
            return -1.0;

        double cell_size = 0.0;

        for (size_t i = 0; i < c; ++i)
        {
            const RotatedRect& box = boxesPattern[i];

            cell_size += box.size.width + box.size.height;
        }

        cell_size /= 42.0;

        return cell_size;
    }

    Mat getCodeMatrix(const Mat& src, const vector<RotatedRect>& boxesPattern)
    {
        double cell_size = getCellSize(boxesPattern);
        if (cell_size <= 0)
            return Mat();

        int version = static_cast<int>((src.cols / cell_size - 17.0) / 4.0);
        int w = 4 * version + 17;

        Mat dst;
        resize(src, dst, Size(w, w));

        return dst;
    }
}

void Qr::posterizeImage(const Mat& image, const vector<RotatedRect>& boxesPattern, Mat& code,
                        int th_low, int th_hi, int th_step, int adaptive_th_size, int adaptive_th_delta)
{
    CV_Assert(image.type() == CV_8UC1);
    CV_Assert(boxesPattern.size() == 3);

    Mat buf;
    code.release();
    for (int th = th_low; th <= th_hi && code.empty(); th += th_step)
    {
        createPosterizedImage(image, buf, adaptive_th_size, adaptive_th_delta * 3, th, DEFAULT_POSTERIZED_TH_HI);

        code = getCodeMatrix(buf, boxesPattern);
    }
}

namespace
{
    Mat processFrame_binarized;
    vector<RotatedRect> processFrame_borderPattern;

    void processFrame(const Mat& frame, Mat& outFrame, bool drawPoints, bool drawAxis, Mat& rvec, Mat& tvec)
    {
        Qr::binarize(frame, processFrame_binarized);

        //
        // find finder patterns from binarized image
        //
        processFrame_borderPattern.clear();
        Qr::findBorderPattern(processFrame_binarized, processFrame_borderPattern);

        if (drawPoints) {
            Scalar colors[] = { Scalar(255, 0, 0), Scalar(0, 255, 0), Scalar(0, 0, 255) };
            for (size_t i = 0; i < processFrame_borderPattern.size(); ++i)
                ellipse(outFrame, processFrame_borderPattern[i], colors[i % 3], 2);
        }

//        vector<RotatedRect> boxesPattern = borderPattern;
//
//        //
//        // find code area from binarized image using finder patterns
//        //
//        vector<cv::Point> codeAreaContour;
//        Qr::findCodeAreaContour(binarized, borderPattern, codeAreaContour);
//
//        //
//        // perspective transform from source image
//        //
//        Mat transformed;
//        transformImage(binarized, transformed, codeAreaContour, boxesPattern);
//
//        //
//        // posterize source image
//        //
//        Mat code;
//        posterizeImage(transformed, boxesPattern, code);
    }
}

extern "C"
{
    JNIEXPORT void JNICALL Java_com_googlecode_ardemo_QrDetector_processFrame(JNIEnv*, jclass,
            jlong frame_, jlong outFrame_, jboolean drawPoints, jboolean drawAxis, jlong rvec_, jlong tvec_)
    {
        Mat* frame = (Mat*)frame_;
        Mat* outFrame = (Mat*)outFrame_;
        Mat* rvec = (Mat*)rvec_;
        Mat* tvec = (Mat*)tvec_;

        processFrame(*frame, *outFrame, drawPoints, drawAxis, *rvec, *tvec);
    }

    JNIEXPORT void JNICALL Java_com_googlecode_ardemo_QrDetector_releaseNative(JNIEnv*, jclass)
    {
        binarizeBuf.release();

        vector< vector<Point> >().swap(findBorderPattern_contours);
        vector< pair< Rect, const vector<Point>* > >().swap(findBorderPattern_candidates);

        processFrame_binarized.release();
        vector<RotatedRect>().swap(processFrame_borderPattern);
    }
}
