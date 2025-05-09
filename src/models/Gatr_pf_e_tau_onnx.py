from os import path
import sys

from time import time
from src.gatr_v111.nets.gatr import GATr
from src.gatr_v111.layers.attention.config import SelfAttentionConfig
from src.gatr_v111.layers.mlp.config import MLPConfig
from gatr.interface import (
    embed_point,
    extract_scalar,
    extract_point,
    embed_scalar,
    embed_translation,
)
from src.gatr.primitives.linear import _compute_pin_equi_linear_basis
from src.gatr.primitives.attention import _build_dist_basis

# from gatr import GATr, SelfAttentionConfig, MLPConfig
# from gatr.interface import embed_point, extract_scalar, extract_point, embed_scalar
from torch_scatter import scatter_add, scatter_mean, scatter_max
import torch
import torch.nn as nn
from src.utils.save_features import save_features
from src.logger.plotting_tools import PlotCoordinates
import numpy as np
from typing import Tuple, Union, List
import dgl
from src.logger.plotting_tools import PlotCoordinates
from src.models.mlp_readout_layer import MLPReadout

import lightning as L
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from src.layers.inference_oc import create_and_store_graph_output
# from xformers.ops.fmha import BlockDiagonalMask
import os
import wandb

import torch.nn.functional as F
import pandas as pd

from sklearn.utils.class_weight import compute_class_weight


class ExampleWrapper(L.LightningModule):
    """Example wrapper around a GATr model.

    Expects input data that consists of a point cloud: one 3D point for each item in the data.
    Returns outputs that consists of one scalar number for the whole dataset.

    Parameters
    ----------
    blocks : int
        Number of transformer blocks
    hidden_mv_channels : int
        Number of hidden multivector channels
    hidden_s_channels : int
        Number of hidden scalar channels
    """

    def __init__(
        self,
        args,
        dev,
    ):
        super().__init__()
        print("using this model")
        self.strict_loading = False

        self.loss_final = 0
        self.number_b = 0
        self.df_showers = []
        self.df_showers_pandora = []
        self.df_showes_db = []
        
        blocks = 10
        hidden_mv_channels = 16
        hidden_s_channels = 64
        self.input_dim = 3
        self.output_dim = 4
        self.args = args
        self.basis_gp = None
        self.basis_outer = None
        self.pin_basis = None
        self.basis_q = None
        self.basis_k = None
        self.args = args
        self.load_basis()
        self.gatr = GATr(
            in_mv_channels=1,
            out_mv_channels=1,
            hidden_mv_channels=hidden_mv_channels,
            in_s_channels=4,
            out_s_channels=1,
            hidden_s_channels=hidden_s_channels,
            num_blocks=blocks,
            attention=SelfAttentionConfig(),  #
            mlp=MLPConfig(),  
            basis_gp=self.basis_gp,
            basis_outer=self.basis_outer,
            basis_pin=self.pin_basis,
            basis_q=self.basis_q,
            basis_k=self.basis_k,
        )
        self.ScaledGooeyBatchNorm2_1 = nn.BatchNorm1d(self.input_dim, momentum=0.1)
        number_of_classes = 4
        self.loss_crit = nn.CrossEntropyLoss()
        self.readout = "sum"
        self.MLP_layer = MLPReadout(17+ 4, number_of_classes)
        self.m = nn.Softmax(dim=1)

    def load_basis(self):
        filename = "/afs/cern.ch/user/m/mgarciam/.local/lib/python3.8/site-packages/gatr/primitives/data/geometric_product.pt"
        sparse_basis = torch.load(filename).to(torch.float32)
        basis = sparse_basis.to_dense()
        self.basis_gp = basis.to(device="cuda:3")
        filename = "/afs/cern.ch/user/m/mgarciam/.local/lib/python3.8/site-packages/gatr/primitives/data/outer_product.pt"
        sparse_basis_outer = torch.load(filename).to(torch.float32)
        sparse_basis_outer = sparse_basis_outer.to_dense()
        self.basis_outer = sparse_basis_outer.to(device="cuda:3")

        self.pin_basis = _compute_pin_equi_linear_basis(
            device=self.basis_gp.device, dtype=basis.dtype
        )
        self.basis_q, self.basis_k = _build_dist_basis(
            device=self.basis_gp.device, dtype=basis.dtype
        )
    def forward(self, inputs, inputs_scalar, energy_inputs, feature_high_level):
        """Forward pass.

        Parameters
        ----------
        inputs : torch.Tensor with shape (*batch_dimensions, num_points, 3)
            Point cloud input data

        Returns
        -------
        outputs : torch.Tensor with shape (*batch_dimensions, 1)
            Model prediction: a single scalar for the whole point cloud.
        """
        inputs = self.ScaledGooeyBatchNorm2_1(inputs)
        embedded_inputs = embed_point(inputs) 
        embedded_inputs = embedded_inputs.unsqueeze(
            -2
        )  # (batch_size*num_points, 1, 16)

        # scalars = torch.zeros((inputs.shape[0], 1))
        scalars = torch.cat(
            (energy_inputs, inputs_scalar), dim=1
        )  

        embedded_outputs, scalar_outputs = self.gatr(
            embedded_inputs, scalars=scalars
        ) 
        h_out = torch.cat(
            (embedded_outputs[:, 0, :], scalar_outputs.view(-1, 1)), dim=1
        )
        hg = torch.sum(h_out, dim=0) 
        all_features = torch.cat((feature_high_level, hg.view(-1, 17)), dim=1)
        all_features = self.MLP_layer(all_features)
        return all_features

    def build_attention_mask(self, g):
        """Construct attention mask from pytorch geometric batch.

        Parameters
        ----------
        inputs : torch_geometric.data.Batch
            Data batch.

        Returns
        -------
        attention_mask : xformers.ops.fmha.BlockDiagonalMask
            Block-diagonal attention mask: within each sample, each token can attend to each other
            token.
        """
        batch_numbers = obtain_batch_numbers_tau(g)

        return (
            0, 
            batch_numbers,
        )

    def obtain_loss_weighted(self, labels_true):
        # class_weights = compute_class_weight(
        #     "balanced",
        #     classes=torch.unique(labels_true).detach().cpu().numpy(),
        #     y=labels_true.detach().cpu().numpy(),
        # )
        w_e = 70 / (17)
        w_mu = 70 / (18)
        w_rho = 70 / (25)
        w_pi = 70 / (10)

        class_weights = torch.tensor([w_e, w_mu, w_rho, w_pi]).to(labels_true.device)
        unique_class_labels = torch.unique(labels_true).long()
        # weights_all_classes = torch.zeros(4).to(unique_class_labels.device)
        # weights_all_classes[unique_class_labels] = torch.Tensor(class_weights).to(
        #     unique_class_labels.device
        # )

        self.loss_crit = nn.CrossEntropyLoss(weight=class_weights[unique_class_labels])
    def obtain_high_level_features(self, g, labels):
        g.ndata["hit_type1"] = 1.0 * (g.ndata["hit_type"] == 1)
        g.ndata["hit_type2"] = 1.0 * (g.ndata["hit_type"] == 2)
        g.ndata["hit_type3"] = 1.0 * (g.ndata["hit_type"] == 3)
        g.ndata["hit_type4"] = 1.0 * (g.ndata["hit_type"] == 4)

        feature_high_level = torch.cat(
            (
                scatter_add(g.ndata["hit_type1"].view(-1), labels.long().view(-1)).view(
                    -1, 1
                ),
                scatter_add(g.ndata["hit_type2"].view(-1), labels.long().view(-1)).view(
                    -1, 1
                ),
                scatter_add(g.ndata["hit_type3"].view(-1), labels.long().view(-1)).view(
                    -1, 1
                ),
                scatter_add(g.ndata["hit_type4"].view(-1), labels.long().view(-1)).view(
                    -1, 1
                ),
            ),
            dim=1,
        )
        return feature_high_level
    
    def training_step(self, batch, batch_idx):
        y = batch[1]
        batch_g = batch[0]
        initial_time = time()
        mask, labels = self.build_attention_mask(batch_g)
        inputs = batch_g.ndata["pos_hits_xyz"]
        inputs_scalar = batch_g.ndata["hit_type"].view(-1, 1)
        energy_inputs = batch_g.ndata["h"][:, -3:]
        feature_high_level = self.obtain_high_level_features(batch_g, labels) 
        # do one evaluation per tau and then add them up

        model_output = self(inputs, inputs_scalar, energy_inputs, feature_high_level)
        
        # Dummy loss to avoid errors
        labels_true = scatter_max(batch_g.ndata["label_true"].view(-1), labels.long())[
            0
        ]
        # self.obtain_loss_weighted(labels_true)
        loss = self.loss_crit(
            model_output,
            labels_true.view(-1).long(),
        )
        loss_time_end = time()
       

        misc_time_start = time()
        if self.trainer.is_global_zero:
            wandb.log({"loss": loss.item()})
            acc = torch.mean(
                1.0 * (model_output.argmax(axis=1) == labels_true.view(-1))
            )
            wandb.log({"accuracy": acc.item()})
        self.loss_final = loss.item() + self.loss_final
        self.number_b = self.number_b + 1
        del model_output

        final_time = time()
        if self.trainer.is_global_zero:
            wandb.log({"misc_time_inside_training": final_time - misc_time_start})
            wandb.log({"training_step_time": final_time - initial_time})
        return loss

    def validation_step(self, batch, batch_idx):
        cluster_features_path = os.path.join(self.args.model_prefix, "cluster_features")
        show_df_eval_path = os.path.join(
            self.args.model_prefix, "showers_df_evaluation"
        )
        if not os.path.exists(show_df_eval_path):
            os.makedirs(show_df_eval_path)
        if not os.path.exists(cluster_features_path):
            os.makedirs(cluster_features_path)
        self.validation_step_outputs = []
        y = batch[1]
        batch_g = batch[0]
        shap_vals, ec_x = None, None
        mask, labels = self.build_attention_mask(batch_g)
        inputs = batch_g.ndata["pos_hits_xyz"]
        inputs_scalar = batch_g.ndata["hit_type"].view(-1, 1)
        energy_inputs = batch_g.ndata["h"][:, -3:]
        feature_high_level = self.obtain_high_level_features(batch_g, labels) 
        model_output = self(inputs, inputs_scalar, energy_inputs, feature_high_level)
        labels_true = scatter_max(batch_g.ndata["label_true"].view(-1), labels.long())[
            0
        ]
        print("saving graph")
        dic = {}
        dic["model_output"] = model_output
        dic["graph"] = batch_g
        dic["part_true"] = y

        torch.save(
            dic,
            self.args.model_prefix + "/graphs_base_secondtau/" + str(batch_idx) + ".pt",
        )
        # create_and_store_graph_output(
        #     labels_true,
        #     batch_g,
        #     model_output,
        #     y,
        #     batch_idx,
        #     path_save=self.args.model_prefix,
        # )
        # self.obtain_loss_weighted(labels_true)
        loss = self.loss_crit(
            model_output,
            labels_true.view(-1).long(),
        )

        model_output1 = model_output
        # print(model_output1.shape, labels_true.shape, y.mom_main_daugther.shape)
        # if self.args.predict:
        #     # d = {
        #     #     "pi": model_output1.detach().cpu()[:, 0].view(-1),
        #     #     # "pi0": model_output1.detach().cpu()[:, 1].view(-1),
        #     #     "e": model_output1.detach().cpu()[:, 1].view(-1),
        #     #     "muon": model_output1.detach().cpu()[:, 2].view(-1),
        #     #     "rho": model_output1.detach().cpu()[:, 3].view(-1),
        #     #     "labels_true": labels_true.detach().cpu().view(-1),
        #     #     # "energy": y.E.detach().cpu().view(-1),
        #     # }
        #     d = {
        #         "e": model_output1.detach().cpu()[:, 0].view(-1),
        #         "muon": model_output1.detach().cpu()[:, 1].view(-1),
        #         "rho": model_output1.detach().cpu()[:, 2].view(-1),
        #         "pi": model_output1.detach().cpu()[:, 3].view(-1),
        #         "labels_true": labels_true.detach().cpu().view(-1),
        #         "mom_p": y.mom_main_daugther.detach().cpu().view(-1),
        #         # "energy": y.E.detach().cpu().view(-1),
        #     }
        #     df = pd.DataFrame(data=d)
        #     self.eval_df.append(df)

        # if self.trainer.is_global_zero:
        # print(model_output)
        # print(labels_true)
        # wandb.log({"loss_val": loss.item()})
        # acc = torch.mean(1.0 * (model_output.argmax(axis=1) == labels_true.view(-1)))
        # # print(acc)
        # wandb.log({"accuracy val ": acc.item()})

        # if self.trainer.is_global_zero:
        # wandb.log(
        #     {
        #         "conf_mat": wandb.plot.confusion_matrix(
        #             probs=None,
        #             y_true=labels_true.view(-1).detach().cpu().numpy(),
        #             preds=model_output.argmax(axis=1).view(-1).detach().cpu().numpy(),
        #             # class_names=["pi", "pi0", "e", "muon", "rho"],
        #             class_names=["e", "muon", "rho", "pi"],
        #         )
        #     }
        # )

        del loss
        del model_output

    def on_train_epoch_end(self):

        self.log("train_loss_epoch", self.loss_final / self.number_b)

    def on_train_epoch_start(self):
        # if self.trainer.is_global_zero and self.current_epoch == 0:
        #     self.stat_dict = {}
        self.make_mom_zero()

    def on_validation_epoch_start(self):
        self.make_mom_zero()
        self.eval_df = []

    # def on_validation_epoch_end(self):
    #     if self.args.predict:
    #         df_batch1 = pd.concat(self.eval_df)
    #         df_batch1.to_pickle(self.args.model_prefix + "/model_output_eval_logits.pt")

    # def on_after_backward(self):
    #     for name, p in self.named_parameters():
    #         if p.grad is None:
    #             print(name)

    def make_mom_zero(self):
        if self.current_epoch > 1 or self.args.predict:
            print("making momentum 0")
            self.ScaledGooeyBatchNorm2_1.momentum = 0

    # def on_validation_epoch_end(self):

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(
        #     filter(lambda p: p.requires_grad, self.parameters()), lr=1e-3
        # )
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # if self.args.lr_scheduler == "cosine":
       
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(20),
            eta_min=0,
        )
        # print("Optimizer params:", filter(lambda p: p.requires_grad, self.parameters()))
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler, #ReduceLROnPlateau(optimizer, patience=3),
                "interval": "epoch",
                "monitor": "train_loss_epoch",
                "frequency": 1
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }


def obtain_batch_numbers_tau(g):
    graphs_eval = dgl.unbatch(g)
    number_graphs = len(graphs_eval)
    batch_numbers = []
    counter_index = 0
    for index in range(0, number_graphs):
        gj = graphs_eval[index]
        number_of_graphs = torch.unique(gj.ndata["tau_label"])
        for el in range(0, len(number_of_graphs)):
            g1_number_of_nodes = torch.sum(
                gj.ndata["tau_label"] == number_of_graphs[el]
            )
            batch_numbers.append(
                counter_index
                * torch.ones(g1_number_of_nodes).to(gj.ndata["tau_label"].device)
            )
            counter_index = counter_index + 1

    batch = torch.cat(batch_numbers, dim=0)
    return batch


def graph_sum_output(labels, h):
    tensor = h
    num_fea = h.shape[-1]
    unique_indices, inverse_indices = torch.unique(labels, return_inverse=True)
    unique_indices = unique_indices.to(h.device)
    inverse_indices = inverse_indices.to(h.device)
    output = torch.zeros((len(unique_indices), num_fea)).to(h.device)
    output.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, num_fea), tensor)
    return output
