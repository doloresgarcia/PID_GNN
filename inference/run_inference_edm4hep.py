import sys
import math
import ROOT
from array import array
from ROOT import TFile, TTree
import numpy as np
from podio import root_io
import edm4hep
import onnxruntime as ort
from tree_tools_tautau_inference import (
    initialize,
    get_tracks,
    store_calo_hits,
    create_inputs, 
    split_taus
)


## global params
CALO_RADIUS_IN_MM = 1500



input_file = sys.argv[1]

reader = root_io.Reader(input_file)

########################################

ort.set_default_logger_severity(0)

so = ort.SessionOptions()
so.enable_profiling = True

so.inter_op_num_threads = 1
so.intra_op_num_threads = 1
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL


ort_session = ort.InferenceSession(
    "/eos/user/m/mgarciam/PID_GNN/Ztautau_220924_multiclass/model_multivector_3_input.onnx",
    sess_options=so,
)

### initiali
for i, event in enumerate(reader.get("events")):
        if i ==1:

            dic = initialize()
            dic = get_tracks(
                event,
                dic
            )
            dic = store_calo_hits(
                event,
                dic
            )

            input_data, inputs = create_inputs(dic)
            input_data1, input_data2 = split_taus(input_data, inputs)
            np.save( "input_data1.npy",input_data1)
            np.save("input_data2.npy",input_data2)
            print(input_data1.shape,input_data2.shape)
            ort_inputs = {
                ort_session.get_inputs()[0].name: np.float32(input_data1),
            }  
            ort_outs1 = ort_session.run(None, ort_inputs)
            print("tau 1 is classified as", ort_outs1[0])


       

            ort_inputs = {
                ort_session.get_inputs()[0].name: np.float32(input_data2),
            }  
            ort_outs2 = ort_session.run(None, ort_inputs)
            print("tau 2 is classified as", ort_outs2[0])