//
// Created by jacekp on 08.12.23.
//

#ifndef MODERNML_SCALER_HPP
#define MODERNML_SCALER_HPP

#include "../utils/Concepts.hpp"
#include "../utils/Outputs.hpp"
#include <boost/numeric/ublas/matrix.hpp>
namespace ublas = boost::numeric::ublas;

namespace ModernML
{
    template<FloatingPoint datatype>
    class Scaler
    {
        virtual ScalerOutputs fit(ublas::matrix<datatype> &matrix);
        virtual ublas::matrix<datatype> transform(ublas::matrix<datatype> &matrix, bool copy);
        virtual ublas::matrix<datatype> fitAndTransform(ublas::matrix<datatype> &matrix, bool copy);
    };
}

#endif //MODERNML_SCALER_HPP
