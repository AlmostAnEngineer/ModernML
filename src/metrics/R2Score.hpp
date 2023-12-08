//
// Created by jacekp on 30.11.23.
//

#ifndef MODERNML_R2SCORE_HPP
#define MODERNML_R2SCORE_HPP

#include <boost/numeric/ublas/matrix.hpp>
#include "../utils/Concepts.hpp"

namespace ublas = boost::numeric::ublas;
namespace ModernML
{
    template<FloatingPoint datatype>
    ublas::matrix<datatype> R2Score(const ublas::matrix<datatype> &yReal, const ublas::matrix<datatype> &yPred)
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
}

#endif //MODERNML_R2SCORE_HPP
