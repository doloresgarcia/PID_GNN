#!/usr/bin/env python

import os
import sys
import argparse
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
from src.utils.parser_args import parser

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import onnxscript
from onnxscript.onnx_opset import opset18 as op
import os


# custom_opset = onnxscript.values.Opset(domain="torch.onnx", version=18)

# @onnxscript.script(custom_opset)
# def ScaledDotProductAttention(
#     query,
#     key,
#     value,
#     dropout_p = 0.0,
#     scale = None,
# ):
#     # Swap the last two axes of key
#     key_shape = op.Shape(key)
#     key_last_dim = key_shape[-1:]
#     key_second_last_dim = key_shape[-2:-1]
#     key_first_dims = key_shape[:-2]
#     # Contract the dimensions that are not the last two so we can transpose
#     # with a static permutation.
#     key_squeezed_shape = op.Concat(
#         op.Constant(value_ints=[-1]), key_second_last_dim, key_last_dim, axis=0
#     )
#     key_squeezed = op.Reshape(key, key_squeezed_shape)
#     key_squeezed_transposed = op.Transpose(key_squeezed, perm=[0, 2, 1])
#     key_transposed_shape = op.Concat(key_first_dims, key_last_dim, key_second_last_dim, axis=0)
#     key_transposed = op.Reshape(key_squeezed_transposed, key_transposed_shape)

#     # https://github.com/pytorch/pytorch/blob/12da0c70378b5be9135c6fda62a9863bce4a4818/aten/src/ATen/native/transformers/attention.cpp#L653
#     # Scale q, k before matmul for stability see https://tinyurl.com/sudb9s96 for math
#     query_scaled = op.Mul(query, op.Sqrt(scale))
#     key_transposed_scaled = op.Mul(key_transposed, op.Sqrt(scale))
#     attn_weight = op.Softmax(
#         op.MatMul(query_scaled, key_transposed_scaled),
#         axis=-1,
#     )
#     attn_weight, _ = op.Dropout(attn_weight, dropout_p)
#     return op.MatMul(attn_weight, value)


# def custom_scaled_dot_product_attention(g, query, key, value, attn_mask, dropout, is_causal, scale):
#     return g.onnxscript_op(ScaledDotProductAttention, query, key, value, dropout, scale).setType(V.type())


# torch.onnx.register_custom_op_symbolic(
#     symbolic_name="aten::scaled_dot_product_attention",
#     symbolic_fn=custom_scaled_dot_product_attention,
#     opset_version=18,
# )





def main():
    args = parser.parse_args()
    args.local_rank = 0
    # torch.onnx.register_custom_op_symbolic(
    # symbolic_name="aten::scaled_dot_product_attention",
    # symbolic_fn=custom_scaled_dot_product_attention,
    # opset_version=18,
    # )

    if args.export_onnx:
        # torch.backends.mkldnn.set_flags(False)
        # torch.backends.nnpack.set_flags(False)
        print("exporting to onnx")
        filepath = args.model_prefix + "model_multivector_3_input.onnx"
        torch._dynamo.config.verbose = True
        if args.load_model_weights is not None:
            from src.models.Gatr_pf_e_tau_onnx2 import ExampleWrapper
            # model = ExampleWrapper( args=args, dev=0)
            model = ExampleWrapper.load_from_checkpoint(
                args.load_model_weights, args=args, dev=0
            )
        input_data= torch.load("/afs/cern.ch/work/m/mgarciam/private/PID_GNN/notebooks/tensor.pt")
        model.eval()
        model.ScaledGooeyBatchNorm2_1.momentum = 0
        print("output example", model(input_data))
       
        args1 = torch.randn((10, 7))
        # features_high_level = torch.cat((torch.randn(10,1), torch.randn(10,3)),dim=1)
        # args1 = (torch.randn((10, 3)), torch.randn((10, 1)), torch.randn((10, 3)), torch.randn((10, 4))) #, torch.randn((10, 1)), torch.randn((10, 3)), features_high_level)
        # args1 = torch.randn((10, 11))
        export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
        onnx_program = torch.onnx.dynamo_export(
            model,  args1, export_options=export_options)
        torch.onnx.export(model, 
                        args1,
                        # inputs_scalar = torch.randn((10, 3)),
                        # energy_inputs = torch.randn((10, 3)), 
                        # feature_high_level = torch.randn((10, 4)), 
                        filepath, 
                        opset_version=18,
                        dynamo=True, 
                        # report=True,
                        #  verify=True,       
                        input_names=["inputs1","inputs_scalar", "energy_inputs", "feature_high_level"], #, "inputs_scalar", "energy_inputs", "feature_high_level"],
                        # output_names= ["output"],
                        output_names=["output"],
                        dynamic_axes={
                            "inputs1": [0], 
                            "inputs_scalar": [0], 
                            "energy_inputs": [0], 
                            "feature_high_level": [0]}) 
            inputs=torch.zeros((10, 3)), inputs_scalar=torch.zeros((10, 1)), energy_inputs=torch.zeros((10, 3)), feature_high_level=features_high_level, export_options=export_options,
        )
        onnx_program.save(filepath)



if __name__ == "__main__":
    main()
