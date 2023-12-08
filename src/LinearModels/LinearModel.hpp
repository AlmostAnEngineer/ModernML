//
// Created by jacekp on 30.11.23.
//

#ifndef MODERNML_LINEARMODEL_HPP
#define MODERNML_LINEARMODEL_HPP

#include "../utils/Outputs.hpp"
#include "../utils/Concepts.hpp"
#include <boost/numeric/ublas/matrix.hpp>
namespace ublas = boost::numeric::ublas;

namespace ModernML
{
template<FloatingPoint datatype>
    class LinearModel
    {
        virtual LinearRegressionOutputs fit(ublas::matrix<datatype> X, ublas::matrix<datatype> y);

        virtual ublas::matrix<datatype> predict(ublas::matrix<datatype> X);
    };
}

#endif //MODERNML_LINEARMODEL_HPP
