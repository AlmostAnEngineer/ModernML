//
// Created by jacekp on 09.10.23.
//
#include "LinearRegression.hpp"

int main()
{
    ublas::matrix<double> X(5, 2);
    ublas::matrix<double> y(5, 1);

    for (size_t i = 0; i < X.size1(); ++i) {
        for (size_t j = 0; j < X.size2(); ++j) {
            X(i, j) = static_cast<double>(i * X.size2() + j + 1);
        }
        y(i, 0) = static_cast<double>(2 * i + 1);
    }

    LinearRegression<double> model;
    model.fit(X, y);
    return 0;
}