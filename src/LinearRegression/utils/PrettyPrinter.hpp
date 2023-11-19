//
// Created by jacekp on 19.11.23.
//

#ifndef MODERNML_PRETTYPRINTER_HPP
#define MODERNML_PRETTYPRINTER_HPP

#include <iostream>
#include <iomanip>
#include <boost/numeric/ublas/matrix.hpp>

template<typename T>
void printMatrix(const boost::numeric::ublas::matrix<T> &mat)
{
    std::cout << "[\n";
    for (std::size_t i = 0; i < mat.size1(); ++i)
    {
        std::cout << " [";
        for (std::size_t j = 0; j < mat.size2(); ++j)
        {
            std::cout << std::setw(3) << mat(i, j);
            if (j < mat.size2() - 1)
            {
                std::cout << ", ";
            }
        }
        std::cout << "]";
        if (i < mat.size1() - 1)
        {
            std::cout << ",";
        }
        std::cout << "\n";
    }
    std::cout << "]\n";
}

#endif //MODERNML_PRETTYPRINTER_HPP
