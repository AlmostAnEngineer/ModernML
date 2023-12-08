//
// Created by jacekp on 19.11.23.
//

#ifndef MODERNML_OUTPUTS_HPP
#define MODERNML_OUTPUTS_HPP

namespace ModernML
{
    enum LinearRegressionOutputs
    {
        SUCCESS_FIT, ERROR_BAD_SIZE, ERROR_DIVERGED
    };

    enum ScalerOutputs
    {
        SUCCESS, ERROR
    };
}

#endif //MODERNML_OUTPUTS_HPP
