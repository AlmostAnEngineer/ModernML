//
// Created by jacekp on 19.11.23.
//
#pragma once
#ifndef MODERNML_RIDGE_HPP
#define MODERNML_RIDGE_HPP

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include "utils/Concepts.hpp"
#include "utils/RandomGenerator.hpp"

namespace ublas = boost::numeric::ublas;

template<FloatingPoint datatype>
class RidgeRegression
{
public:
    enum RegressionOutputs
    {
        SUCCESS_FIT, ERROR_BAD_SIZE, ERROR_DIVERGED
    };
    explicit RidgeRegression(float learningRate = 0.01, const float alpha = 0.1, unsigned int iterations = 1000,
                             double epsilon = 1E-8):
    learningRate(learningRate),
    alpha(alpha),
    iterations(iterations),
    epsilon(epsilon)
    {}
    ~RidgeRegression() = default;

    [[maybe_unused]] RegressionOutputs fit(){};
private:
    float learningRate;
    float alpha;
    unsigned int iterations;
    double epsilon;
};

#endif //MODERNML_RIDGE_HPP
