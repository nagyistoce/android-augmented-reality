LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

OPENCV_CAMERA_MODULES:=off
include ../../OpenCV-2.3.1/share/OpenCV/OpenCV.mk

LOCAL_MODULE := ARDemo

LOCAL_SRC_FILES := CirclesDetector.cpp QrDetector.cpp

LOCAL_LDLIBS += -llog -ldl

include $(BUILD_SHARED_LIBRARY)
