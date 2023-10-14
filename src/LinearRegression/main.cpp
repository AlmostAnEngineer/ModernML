//
// Created by jacekp on 09.10.23.
//
#include "LinearRegression.hpp"

int main()
{
    ublas::matrix<double> X(100, 2);
    ublas::matrix<double> y(100, 2);

    for (size_t i = 0; i < X.size1(); ++i)
    {
        auto dbl = static_cast<double>(i);
        X(i, 0) = dbl * 1.0;
        X(i, 1) = dbl * 2.0;
        y(i, 0) = dbl * 2.0 + 5.0;
        y(i, 1) = dbl * 3.0 + 3.25;
    }

    LinearRegression<double> model(1e-5, 1000, 1E-3);
    model.fit(X, y);
    return 0;
}