#pragma once
#ifndef MODERNML_LINEARREGRESSION_H
#define MODERNML_LINEARREGRESSION_H

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include "utils/Concepts.hpp"
#include "utils/RandomGenerator.hpp"
#include "utils/Outputs.hpp"


namespace ublas = boost::numeric::ublas;

template<FloatingPoint datatype>
class LinearRegression
{
public:
    explicit LinearRegression(float learningRate = 0.01, unsigned int iterations = 1000, double epsilon = 1E-8) :
            readyToPredict(false),
            learningRate(learningRate),
            maxIterations(iterations),
            epsilon(epsilon),
            num_features(0),
            num_outputs(0)
    {};

    ~LinearRegression() = default;

    [[maybe_unused]] LinearRegressionOutputs fit(ublas::matrix<datatype> X, ublas::matrix<datatype> y)
    {
        if (X.size1() != y.size1())
        {
            return LinearRegressionOutputs::ERROR_BAD_SIZE;
        }

        const auto num_samples = X.size1();
        num_features = X.size2();
        num_outputs = y.size2();

        ublas::matrix<datatype> coefficients;
        coefficients.resize(1, num_features, true);

        ublas::matrix<datatype> biases(num_outputs, 1, 1);
        ublas::matrix<datatype> all_weights(num_outputs, num_features, getRandomNumber<datatype>(-1, 1));

        for (auto columnIndex = 0; columnIndex < num_outputs; ++columnIndex)
        {
            datatype loss = 0.0;
            datatype error;

            ublas::vector<datatype> weights = ublas::row(all_weights, columnIndex);
            datatype bias = getRandomNumber<datatype>(-1, 1);

            for (auto iteration = 0; iteration < maxIterations; ++iteration)
            {
                ublas::vector<datatype> gradient(num_features, 0.0);
                for (auto sample = 0; sample < num_samples; ++sample)
                {
                    ublas::matrix_row<ublas::matrix<datatype>> x_sample(X, sample);
                    auto prediction = ublas::inner_prod(x_sample, weights) + bias;
                    error = prediction - y(sample, columnIndex);
                    loss += error * error;
                    gradient += 2 * error * x_sample;
                }
                loss /= num_samples;
                gradient /= num_samples;

                weights -= learningRate * gradient;
                bias -= learningRate * loss;

                if (loss < epsilon)
                    break;

                if (std::isnan(loss) || std::isinf(loss))
                {
                    return LinearRegressionOutputs::ERROR_DIVERGED;
                }
            }

            ublas::row(all_weights, columnIndex) = weights;
            biases(columnIndex, 0) = bias;
        }

        weightsAfterFit = all_weights;
        biasesAfterFit = biases;
        readyToPredict = true;
        return LinearRegressionOutputs::SUCCESS_FIT;
    }


    ublas::matrix<datatype> predict(ublas::matrix<datatype> X)
    {
        if (X.size2() != num_features || !readyToPredict)
        {
            return {0, 0};
        }

        const auto num_samples = X.size1();
        ublas::matrix<datatype> predictions(num_samples, num_outputs, 0.0);

        for (auto sample = 0; sample < num_samples; ++sample)
        {
            for (auto columnIndex = 0; columnIndex < num_outputs; ++columnIndex)
            {
                ublas::matrix_row<ublas::matrix<datatype>> x_sample(X, sample);
                datatype prediction = ublas::inner_prod(x_sample, ublas::row(weightsAfterFit, columnIndex)) +
                                      biasesAfterFit(columnIndex, 0);
                predictions(sample, columnIndex) = prediction;
            }
        }
        return predictions;
    }

    ublas::matrix<datatype> getR2Scores(const ublas::matrix<datatype> &yReal, const ublas::matrix<datatype> &yPred)
    {
        if (yReal.size1() != yPred.size1() || yReal.size2() != yPred.size2() || yReal.size2() == 0)
        {
            return ublas::matrix<datatype>(1, 1, std::numeric_limits<datatype>::quiet_NaN());
        }

        const size_t numColumns = yReal.size2();
        ublas::matrix<datatype> r2Scores(1, numColumns, 0.0);

        for (size_t col = 0; col < numColumns; ++col)
        {
            datatype yMean = 0.0;
            for (size_t i = 0; i < yReal.size1(); ++i)
            {
                yMean += yReal(i, col);
            }
            yMean /= yReal.size1();

            datatype ssr = 0.0;
            datatype sse = 0.0;

            for (size_t i = 0; i < yReal.size1(); ++i)
            {
                ssr += (yPred(i, col) - yMean) * (yPred(i, col) - yMean);
                sse += (yReal(i, col) - yPred(i, col)) * (yReal(i, col) - yPred(i, col));
            }

            if (ssr == 0.0)
            {
                r2Scores(0, col) = 1.0;
            } else
            {
                r2Scores(0, col) = 1.0 - sse / ssr;
            }
        }

        return r2Scores;
    }


private:
    float learningRate;
    unsigned int maxIterations;
    double epsilon;
    bool readyToPredict;

    ublas::matrix<datatype> biasesAfterFit;
    ublas::matrix<datatype> weightsAfterFit;
    size_t num_features;
    size_t num_outputs;
};

#endif // MODERNML_LINEARREGRESSION_H
