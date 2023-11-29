//
// Created by jacekp on 29.11.23.
//

#include "LinearModels/LinearRegression.hpp"
#include <iostream>
#include <boost/numeric/ublas/matrix.hpp>

namespace ublas = boost::numeric::ublas;

int main()
{
    auto X =  ublas::matrix<double>(300,3, 2.0);
    auto y =  ublas::matrix<double>(300,1, 1.0);
    auto model = LinearRegression<double>();
    auto out = model.fit(X,y);
    auto pred = model.predict(X);
    std::cout << model.getR2Scores(y, pred)(0,0);
}
