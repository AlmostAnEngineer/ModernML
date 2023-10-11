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
#include <boost/numeric/ublas/matrix_proxy.hpp>
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
    enum optimList{GRADIENTDESCENT, SGD, ADAM};
    explicit LinearRegression(float learningRate=0.01, unsigned int iterations=1000, double epsilon=1E-8):
    learningRate(learningRate), iterations(iterations), epsilon(epsilon), optimizator(optimList::GRADIENTDESCENT),\
    adamParams(adamSettings{0.999,0.9,0})
    {};

    ~LinearRegression()=default;

    [[maybe_unused]] void fit(ublas::matrix<datatype> X, ublas::matrix<datatype> y)
    {
        if(X.size1() != y.size1())
            throw std::runtime_error("ERROR");

        const auto num_samples = X.size1();
        const auto num_features = X.size2();

        coefficients_.resize(1, num_features, false);
        weights.resize(num_features, 0.0);
        bias = 0.0;

        if(optimizator== optimList::GRADIENTDESCENT)
        {
            datatype loss = 0.0;
            datatype error;
            datatype prediction;
            for (unsigned int iteration = 0; iteration < iterations; ++iteration) {
                ublas::vector<datatype> gradient(num_features, 0.0);
                for (size_t sample = 0; sample < num_samples; ++sample) {
                    ublas::matrix_row<ublas::matrix<datatype>> x_sample(X, sample);
                    prediction = ublas::inner_prod(x_sample, weights) + bias;
                    error = prediction - y(sample, 0);
                    loss += error * error;
                    gradient += 2 * error * x_sample;
                }
                loss /= num_samples;
                gradient /= num_samples;

                weights -= learningRate * gradient;
                bias -= learningRate * loss;

                if (loss < epsilon) {
                    break;
                }
            }
            std::cout << error;
        }
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
    optimList optimizator;
    float learningRate;
    unsigned int iterations;
    double epsilon;

    ublas::matrix<datatype> X_;
    ublas::matrix<datatype> y_;
    ublas::matrix<datatype> coefficients_;

    datatype bias=0.0;
    ublas::vector<datatype> weights;

    [[maybe_unused]] adamSettings adamParams;
};

#endif //MODERNML_LINEARREGRESSION_H
