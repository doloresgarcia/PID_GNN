
import os
import sys
import numpy as np
import torch

import os
from sklearn.cluster import KMeans
import numpy as np
import onnxruntime as ort



# dic= torch.load("/eos/user/m/mgarciam/PID_GNN/Ztautau_220924_multiclass/graphs_base_v111_4/1_previous.pt", map_location='cpu')



# inputs = dic["graph"].ndata["pos_hits_xyz"]
# inputs_scalar = dic["graph"].ndata["hit_type"].view(-1, 1)
# energy_inputs = dic["graph"].ndata["h"][:, -3:]
# hit_type1 = 1.0 * (dic["graph"].ndata["hit_type"] == 1)
# hit_type2= 1.0 * (dic["graph"].ndata["hit_type"] == 2)
# hit_type3 = 1.0 * (dic["graph"].ndata["hit_type"] == 3)
# hit_type4 = 1.0 * (dic["graph"].ndata["hit_type"] == 4)
# features_high_level = torch.cat((hit_type1.view(-1,1),hit_type2.view(-1,1), hit_type3.view(-1,1),hit_type4.view(-1,1)), dim=1)
        
# input_data = torch.cat((inputs, inputs_scalar, energy_inputs,features_high_level ), dim=1)

inputs = # np array with x,y,z position of hits [N,3]  (ECAL hits, HCAL hits, muon hits, tracks (no track hits))
inputs_scalar = # np array with hit type of hits [N,1] (1..4)
energy_inputs = # np array with size [N,3], e hits (0 if track), p hits (0 if not track), log(e)
hit_type1 = 1.0 * (inputs_scalar == 1)
hit_type2= 1.0 * (inputs_scalar == 2)
hit_type3 = 1.0 * (inputs_scalar == 3)
hit_type4 = 1.0 * (inputs_scalar == 4)
features_high_level = np.concatenate((hit_type1.view(-1,1),hit_type2.view(-1,1), hit_type3.view(-1,1),hit_type4.view(-1,1)), axis=1)
input_data = np.concatenate((inputs, inputs_scalar, energy_inputs,features_high_level),axis=1)



X = inputs
kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)
labels = kmeans.labels_
mask_tau1 = labels==0
mask_tau2 = labels==1

input_data1 = input_data[mask_tau1]
input_data2 = input_data[mask_tau2]
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


input_data1 = input_data1.cpu().numpy()

ort_inputs = {
    ort_session.get_inputs()[0].name: input_data1,
}  
ort_outs1 = ort_session.run(None, ort_inputs)
print("tau 1 is classified as", ort_outs1[0])


input_data2 = input_data2.cpu().numpy()

ort_inputs = {
    ort_session.get_inputs()[0].name: input_data2,
}  
ort_outs2 = ort_session.run(None, ort_inputs)
print("tau 2 is classified as", ort_outs2[0])