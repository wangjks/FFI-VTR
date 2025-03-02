#ifndef ONNX_HELPER_H
#define ONNX_HELPER_H
#include <onnxruntime_cxx_api.h>
#include <iostream>

bool initOrtSession(Ort::Env& env, Ort::Session& session, std::string& modelPath, const int& gpuId);

#endif