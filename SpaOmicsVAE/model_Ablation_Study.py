import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class Encoder_overall(Module):
    """Overall encoder with ablation control."""
    def __init__(self, dim_in_feat_omics1, dim_out_feat_omics1, dim_in_feat_omics2, dim_out_feat_omics2,
                 use_spatial=True,  # Control whether to use spatial graphs
                 use_feature=True,  # Control whether to use feature graphs
                 use_vae=True,      # Control whether to use Variational Autoencoder
                 use_attention=False, # Control whether to use cross-modality attention
                 dropout=0.0, act=F.relu):
        super(Encoder_overall, self).__init__()
        
        # Model dimensions
        self.dim_in_feat_omics1 = dim_in_feat_omics1
        self.dim_in_feat_omics2 = dim_in_feat_omics2
        self.dim_out_feat_omics1 = dim_out_feat_omics1
        self.dim_out_feat_omics2 = dim_out_feat_omics2
        
        # Ablation controls
        self.use_spatial = use_spatial
        self.use_feature = use_feature
        self.use_vae = use_vae
        self.use_attention = use_attention
        
        # Activation and dropout
        self.dropout = dropout
        self.act = act

        # Encoders and decoders
        self.encoder_omics1 = Encoder(self.dim_in_feat_omics1, self.dim_out_feat_omics1, dropout=self.dropout, act=self.act)
        self.decoder_omics1 = Decoder(self.dim_out_feat_omics1, self.dim_in_feat_omics1, dropout=self.dropout, act=self.act)
        self.encoder_omics2 = Encoder(self.dim_in_feat_omics2, self.dim_out_feat_omics2, dropout=self.dropout, act=self.act)
        self.decoder_omics2 = Decoder(self.dim_out_feat_omics2, self.dim_in_feat_omics2, dropout=self.dropout, act=self.act)

        # Attention layer
        if use_attention:
            self.atten_cross = AttentionLayer(self.dim_out_feat_omics1, self.dim_out_feat_omics2, dropout=self.dropout, act=self.act)

        # Learnable weights for spatial and feature graphs
        if use_spatial:
            self.weight_spatial_omics1 = Parameter(torch.FloatTensor(1))
            self.weight_spatial_omics2 = Parameter(torch.FloatTensor(1))
        if use_feature:
            self.weight_feature_omics1 = Parameter(torch.FloatTensor(1))
            self.weight_feature_omics2 = Parameter(torch.FloatTensor(1))

        # VAE components
        if use_vae:
            self.enc_mu_omics1 = nn.Linear(self.dim_out_feat_omics1, self.dim_out_feat_omics1)
            self.enc_log_var_omics1 = nn.Linear(self.dim_out_feat_omics1, self.dim_out_feat_omics1)
            self.enc_mu_omics2 = nn.Linear(self.dim_out_feat_omics2, self.dim_out_feat_omics2)
            self.enc_log_var_omics2 = nn.Linear(self.dim_out_feat_omics2, self.dim_out_feat_omics2)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize learnable parameters."""
        if self.use_spatial:
            nn.init.constant_(self.weight_spatial_omics1, 0.5)
            nn.init.constant_(self.weight_spatial_omics2, 0.5)
        if self.use_feature:
            nn.init.constant_(self.weight_feature_omics1, 0.5)
            nn.init.constant_(self.weight_feature_omics2, 0.5)

    def reparameterize(self, mu, log_var):
        """Reparameterization trick for Variational Autoencoder."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, features_omics1, features_omics2, adj_spatial_omics1, adj_feature_omics1,
                adj_spatial_omics2, adj_feature_omics2):
        """Forward pass with ablation controls."""
        # Spatial graph encoding
        if self.use_spatial:
            emb_latent_spatial_omics1 = self.encoder_omics1(features_omics1, adj_spatial_omics1)
            emb_latent_spatial_omics2 = self.encoder_omics2(features_omics2, adj_spatial_omics2)
        else:
            emb_latent_spatial_omics1 = emb_latent_spatial_omics2 = None

        # Feature graph encoding
        if self.use_feature:
            emb_latent_feature_omics1 = self.encoder_omics1(features_omics1, adj_feature_omics1)
            emb_latent_feature_omics2 = self.encoder_omics2(features_omics2, adj_feature_omics2)
        else:
            emb_latent_feature_omics1 = emb_latent_feature_omics2 = None

        # Combine spatial and feature graphs
        if self.use_spatial and self.use_feature:
            emb_latent_omics1 = self.weight_spatial_omics1 * emb_latent_spatial_omics1 + \
                                self.weight_feature_omics1 * emb_latent_feature_omics1
            emb_latent_omics2 = self.weight_spatial_omics2 * emb_latent_spatial_omics2 + \
                                self.weight_feature_omics2 * emb_latent_feature_omics2
        elif self.use_spatial:
            emb_latent_omics1, emb_latent_omics2 = emb_latent_spatial_omics1, emb_latent_spatial_omics2
        elif self.use_feature:
            emb_latent_omics1, emb_latent_omics2 = emb_latent_feature_omics1, emb_latent_feature_omics2
        else:
            raise ValueError("Both spatial and feature graphs are disabled.")

        # Cross-modality attention
        if self.use_attention:
            emb_latent_combined, alpha_omics_1_2 = self.atten_cross(emb_latent_omics1, emb_latent_omics2)
        else:
            emb_latent_combined = (emb_latent_omics1 + emb_latent_omics2) / 2
            alpha_omics_1_2 = None

        # VAE reparameterization (only if VAE is enabled)
        if self.use_vae:
            mu_omics1 = self.enc_mu_omics1(emb_latent_omics1)
            log_var_omics1 = self.enc_log_var_omics1(emb_latent_omics1)
            mu_omics2 = self.enc_mu_omics2(emb_latent_omics2)
            log_var_omics2 = self.enc_log_var_omics2(emb_latent_omics2)

            emb_latent_omics1 = self.reparameterize(mu_omics1, log_var_omics1)
            emb_latent_omics2 = self.reparameterize(mu_omics2, log_var_omics2)
        else:
            mu_omics1 = log_var_omics1 = mu_omics2 = log_var_omics2 = None
        # Reconstruction
        emb_recon_omics1 = self.decoder_omics1(emb_latent_combined, adj_spatial_omics1)
        emb_recon_omics2 = self.decoder_omics2(emb_latent_combined, adj_spatial_omics2)

        # Cross-reconstruction
        emb_latent_omics1_across_recon = self.encoder_omics2(
            self.decoder_omics2(emb_latent_omics1, adj_spatial_omics2), adj_spatial_omics2)
        emb_latent_omics2_across_recon = self.encoder_omics1(
            self.decoder_omics1(emb_latent_omics2, adj_spatial_omics1), adj_spatial_omics1)

        # Results
        results = {
            'emb_latent_omics1': emb_latent_omics1,
            'emb_latent_omics2': emb_latent_omics2,
            'emb_latent_combined': emb_latent_combined,
            'emb_recon_omics1': emb_recon_omics1,
            'emb_recon_omics2': emb_recon_omics2,
            'emb_latent_omics1_across_recon': emb_latent_omics1_across_recon,
            'emb_latent_omics2_across_recon': emb_latent_omics2_across_recon,
            'alpha': alpha_omics_1_2,
        }
        if self.use_vae:
            results.update({
                'mu_omics1': mu_omics1,
                'log_var_omics1': log_var_omics1,
                'mu_omics2': mu_omics2,
                'log_var_omics2': log_var_omics2
            })

        return results


class Encoder(Module):
    """Modality-specific GNN encoder."""
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Encoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act

        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)
        return x


class Decoder(Module):
    """Modality-specific GNN decoder."""
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Decoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act

        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)
        return x



class AttentionLayer(Module):
    
    """\
    Attention layer.

    Parameters
    ----------
    in_feat: int
        Dimension of input features.
    out_feat: int
        Dimension of output features.     

    Returns
    -------
    Aggregated representations and modality weights.

    """
    
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(AttentionLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        
        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)
        
    def forward(self, emb1, emb2):
        emb = []
        emb.append(torch.unsqueeze(torch.squeeze(emb1), dim=1))
        emb.append(torch.unsqueeze(torch.squeeze(emb2), dim=1))
        self.emb = torch.cat(emb, dim=1)
        
        self.v = F.tanh(torch.matmul(self.emb, self.w_omega))
        self.vu=  torch.matmul(self.v, self.u_omega)
        self.alpha = F.softmax(torch.squeeze(self.vu) + 1e-6)  
        
        emb_combined = torch.matmul(torch.transpose(self.emb,1,2), torch.unsqueeze(self.alpha, -1))
    
        return torch.squeeze(emb_combined), self.alpha    
