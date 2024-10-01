import sys
import math
import ROOT
from array import array
from ROOT import TFile, TTree
import numpy as np
from podio import root_io
import edm4hep
from sklearn.cluster import KMeans
import numpy as np
c_light = 2.99792458e8
Bz_clic = 4.0
Bz_cld = 2.0
mchp = 0.139570



def omega_to_pt(omega, isclic):
    if isclic:
        Bz = Bz_clic
    else:
        Bz = Bz_cld
    a = c_light * 1e3 * 1e-15
    return a * Bz / abs(omega)


def track_momentum(trackstate, isclic=True):
    pt = omega_to_pt(trackstate.omega, isclic)
    phi = trackstate.phi
    pz = trackstate.tanLambda * pt
    px = pt * math.cos(phi)
    py = pt * math.sin(phi)
    p = math.sqrt(px * px + py * py + pz * pz)
    energy = math.sqrt(p * p + mchp * mchp)
    theta = math.acos(pz / p)
    # print(p, theta, phi, energy)
    return p, theta, phi, energy, px, py, pz





def initialize():
    
    dic = {
        "hit_x": [],
        "hit_y": [],
        "hit_z": [],
        "hit_e": [],
        "hit_p": [],
        "hit_loge": [],
        "hit_type": [],
    }
    return dic





def get_tracks(
    event,
    dic 
):
    isclic = False
    tracks = ("SiTracks_Refitted", 45)
    SiTracksMCTruthLink = "SiTracksMCTruthLink"
    pandora_pfo = "PandoraPFOs"
    pandora_pfos_event = event.get(pandora_pfo)
    gen_track_link_indextr = event.get(SiTracksMCTruthLink)

    track_coll = tracks[0]
    for j, track in enumerate(event.get(track_coll)):
   
        trackstate = track.getTrackStates()[3]
        referencePoint = trackstate.referencePoint
        x = referencePoint.x
        y = referencePoint.y
        z = referencePoint.z
        R = math.sqrt(x**2 + y**2)
        r = math.sqrt(x**2 + y**2 + z**2)

        chi_s = track.getChi2()
        dic["hit_x"].append(x)
        dic["hit_y"].append(y)
        dic["hit_z"].append(z)
        track_mom = track_momentum(trackstate, isclic=isclic)

        dic["hit_p"].append(track_mom[0])
        dic["hit_e"].append(0)
        dic["hit_loge"].append(np.log(track_mom[0]))
        dic["hit_type"].append(1)  # 0 for tracks at vertex

       
    return dic


def store_calo_hits(
    event,
    dic
):
    ## calo stuff
    ecal_barrel = ("ECALBarrel", 46)
    ecal_endcap = ("ECALEndcap", 47)

    hcal_barrel = ("HCALBarrel", 49)
    hcal_endcap = ("HCALEndcap", 50)
    hcal_other = ("HCALOther", 51)
    gen_calo_links0 = "CalohitMCTruthLink"
    pandora_clusters = "PandoraClusters"
    pandora_pfo = "PandoraPFOs"
    gen_calohit_link_indexhit = event.get(gen_calo_links0)
    pandora_clusters_event = event.get(pandora_clusters)
    pandora_pfos_event = event.get(pandora_pfo)
   
    calohit_collections = [
        ecal_barrel[0],
        hcal_barrel[0],
        ecal_endcap[0],
        hcal_endcap[0],
        hcal_other[0],
        "MUON"
    ]

    for calohit_col_index, calohit_coll in enumerate(calohit_collections):
        for j, calohit in enumerate(event.get(calohit_coll)):
            # check if it belongs to tau i
            hit_collection = calohit.getObjectID().collectionID
            position = calohit.getPosition()
            x = position.x
            y = position.y
            z = position.z
            R = math.sqrt(x**2 + y**2)
            r = math.sqrt(x**2 + y**2 + z**2)

            dic["hit_x"].append(x)
            dic["hit_y"].append(y)
            dic["hit_z"].append(z)
            dic["hit_p"].append(0)
            dic["hit_e"].append(calohit.getEnergy())
            dic["hit_loge"].append(np.log(calohit.getEnergy()))
          
            htype = 2  # 2 if ECAL, 3 if HCAL
            if "HCAL" in calohit_coll:
                htype = 3
            elif  "MUON" in calohit_coll:
                htype = 4

            dic["hit_type"].append(htype)  # 0 for calo hits

    return dic


def create_inputs(dic ):
    dic = convert_dic_to_array(dic)
    inputs = np.concatenate((dic["hit_x"],dic["hit_y"],dic["hit_z"]), axis = 1) # np array with x,y,z position of hits [N,3]  (ECAL hits, HCAL hits, muon hits, tracks (no track hits))
    inputs_scalar = dic["hit_type"]# np array with hit type of hits [N,1] (1..4)
    energy_inputs = np.concatenate((dic["hit_e"],dic["hit_p"],dic["hit_loge"]), axis = 1) # np array with size [N,3], e hits (0 if track), p hits (0 if not track), log(e)
    hit_type1 = 1.0 * (inputs_scalar == 1)
    hit_type2= 1.0 * (inputs_scalar == 2)
    hit_type3 = 1.0 * (inputs_scalar == 3)
    hit_type4 = 1.0 * (inputs_scalar == 4)
    features_high_level = np.concatenate((hit_type1,hit_type2, hit_type3,hit_type4), axis=1)
    input_data = np.concatenate((inputs, inputs_scalar, energy_inputs,features_high_level),axis=1)
    return input_data, inputs
def convert_dic_to_array(dic):
    for k in dic.keys():
        dic_u = np.array(dic[k])
        dic[k] = np.reshape(dic_u, (-1,1))
    return dic 

def split_taus(input_data, inputs):
    X = inputs
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)
    labels = kmeans.labels_
    mask_tau1 = labels==0
    mask_tau2 = labels==1

    input_data1 = input_data[mask_tau1]
    input_data2 = input_data[mask_tau2]
    return input_data1, input_data2