#ifndef XFEAT_H
#define XFEAT_H
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>

class XFeat{
public:
    XFeat();
	int detectAndCompute(const cv::Mat &image, cv::Mat &mkpts, cv::Mat& feats, cv::Mat& sc);
	int match(const cv::Mat& mkpts0, const cv::Mat& feats0, const cv::Mat& mkpts1, const cv::Mat& feats1, std::vector<cv::DMatch>& matches, float minScore, std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2);
	int match_self(const cv::Mat& mkpts0, const cv::Mat& feats0, const cv::Mat& mkpts1, const cv::Mat& feats1, std::vector<cv::DMatch>& matches, float minScore, std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2);
	int match_reshaped(const cv::Mat& mkpts0, const cv::Mat& feats0, const cv::Mat& mkpts1, const cv::Mat& feats1, std::vector<cv::DMatch>& matches, float minScore, std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2);
	int matchlightglue(const cv::Mat& mkpts0, const cv::Mat& feats0, const cv::Mat& sc0, const cv::Mat& mkpts1, const cv::Mat& feats1, cv::Mat& matches, cv::Mat& batch_indexes);

	~XFeat();

	// gpu id
	int gpuId_ = 0;

	// onnxruntime
	Ort::Env env_{ nullptr };
	Ort::Session xfeatSession_{ nullptr };
	Ort::Session matchingSession_{ nullptr };
	Ort::AllocatorWithDefaultOptions allocator;

	//
	std::vector<const char*> xfeatInputNames = { "images" };
	std::vector<const char*> xfeatOutputNames = { "mkpts", "feats", "sc" };
	std::vector<const char*> matchingInputNames = { "mkpts0", "feats0", "sc0", "mkpts1", "feats1"};
	std::vector<const char*> matchingOutputNames = { "matches", "batch_indexes" };

	bool initFinishedFlag_ = false;
};



#endif