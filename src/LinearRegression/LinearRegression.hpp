//
// Created by jacekp on 07.10.23.
//
#pragma once
#ifndef MODERNML_LINEARREGRESSION_H
#define MODERNML_LINEARREGRESSION_H

#include <concepts>
#include <string>
#include <stdexcept>
#include <sstream>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <iostream>

template <typename datatype>
concept FloatingPoint = std::is_floating_point_v<datatype>;

namespace
{
    struct adamSettings
    {
        [[maybe_unused]] float beta1=0.9;
        [[maybe_unused]] float beta2=0.999;
        [[maybe_unused]] float t=0;
    };
}

namespace ublas = boost::numeric::ublas;

template <FloatingPoint datatype>
class LinearRegression {

public:
    explicit LinearRegression(float learningRate=0.01, unsigned int iterations=1000, double epsilon=1E-8):
    learningRate(learningRate), iterations(iterations), epsilon(epsilon)
    {};

    ~LinearRegression()=default;

    [[maybe_unused]] void fit(ublas::matrix<datatype> X, ublas::matrix<datatype> y)
    {

    }

    [[maybe_unused]] datatype getR2Score()
    {

    }

    [[maybe_unused]] void setSGD()
    {
        optimizator = "SGD";
    }

    [[maybe_unused]] void setAdam(adamSettings &AdamOptimSettings)
    {
        optimizator = "Adam";
        adamParams = AdamOptimSettings;
    }

    [[maybe_unused]] void setGradientDescent()
    {
        optimizator = "GradientDescent";
    }

private:
    std::string optimizator="GradientDescent";
    float learningRate;
    unsigned int iterations;
    double epsilon;

    ublas::matrix<datatype> X_;
    ublas::matrix<datatype> y_;
    ublas::matrix<datatype> coefficients_;

    datatype bias=0.0;
    ublas::vector<datatype> weights;

    [[maybe_unused]] adamSettings adamParams{0.9, 0.999, 0};
};

#endif //MODERNML_LINEARREGRESSION_H
