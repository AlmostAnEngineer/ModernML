//
// Created by jacekp on 29.11.23.
//

#ifndef MODERNML_MINMAXSCALER_H
#define MODERNML_MINMAXSCALER_H

#include "Scaler.hpp"
#include "../utils/Concepts.hpp"
#include "../utils/Outputs.hpp"
#include <boost/numeric/ublas/matrix.hpp>
namespace ublas = boost::numeric::ublas;

namespace ModernML
{
    template<FloatingPoint datatype>
    class MinMaxScaler: public Scaler<datatype>
    {
    public:
        MinMaxScaler(bool copy):
        copy(copy),
        trained(false)
        {
        }

        ScalerOutputs fit(ublas::matrix<datatype> &matrix)
        {

        }

        ublas::matrix<datatype> transform(ublas::matrix<datatype> &matrix, bool copy)
        {

        }

        ublas::matrix<datatype> fitAndTransform(ublas::matrix<datatype> &matrix, bool copy)
        {

        }

    private:
        bool copy;
        bool trained;
    };
}

#endif //MODERNML_MINMAXSCALER_H
