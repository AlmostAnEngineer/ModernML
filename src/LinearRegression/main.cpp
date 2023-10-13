//
// Created by jacekp on 09.10.23.
//
#include "LinearRegression.hpp"

int main()
{
    ublas::matrix<double> X(10, 2);
    ublas::matrix<double> y(10, 2);

    for (size_t i = 0; i < X.size1(); ++i)
    {
        for (size_t j = 0; j < X.size2(); ++j)
        {
            X(i, j) = static_cast<double>(i * X.size2() + j + 1);
            y(i, j) = static_cast<double>(i*2 + j*5);
        }
    }

    LinearRegression<double> model(0.001, 1000, 1E-8);
    model.fit(X, y);
    return 0;
}