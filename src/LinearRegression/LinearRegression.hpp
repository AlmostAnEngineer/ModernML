//
// Created by jacekp on 07.10.23.
//

#ifndef MODERNML_LINEARREGRESSION_H
#define MODERNML_LINEARREGRESSION_H

#include <string>

namespace
{
    struct adamSettings
    {
        float beta1;
        float beta2;
        float t;
    };
}

class LinearRegression {

public:
    explicit LinearRegression(float learningRate=0.01, unsigned int iterations=1000, double epsilon=1E-8):
    learningRate(learningRate), iterations(iterations), epsilon(epsilon){};
    ~LinearRegression()=default;
    bool setSGD();
    bool setAdam(adamSettings adamSettings);
    bool setGradientDescent();

protected:
    float learningRate;
    unsigned int iterations;
    double epsilon;

};


#endif //MODERNML_LINEARREGRESSION_H
