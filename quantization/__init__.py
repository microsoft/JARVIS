#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from .quantizer import Quantizer
from .range_linear import RangeLinearQuantWrapper, RangeLinearQuantParamLayerWrapper, PostTrainLinearQuantizer, \
    QuantAwareTrainRangeLinearQuantizer, add_post_train_quant_args, NCFQuantAwareTrainQuantizer, \
    RangeLinearQuantConcatWrapper, RangeLinearQuantEltwiseAddWrapper, RangeLinearQuantEltwiseMultWrapper, ClipMode, \
    RangeLinearEmbeddingWrapper, RangeLinearFakeQuantWrapper, RangeLinearQuantMatmulWrapper
from .clipped_linear import LinearQuantizeSTE, ClippedLinearQuantization, WRPNQuantizer, DorefaQuantizer, PACTQuantizer
from .q_utils import *
from .pytorch_quant_conversion import convert_distiller_ptq_model_to_pytorch, distiller_qparams_to_pytorch, \
    distiller_quantized_tensor_to_pytorch

del quantizer
del range_linear
del clipped_linear
