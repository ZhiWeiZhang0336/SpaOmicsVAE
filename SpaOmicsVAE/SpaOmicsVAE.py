import torch
from tqdm import tqdm
import torch.nn.functional as F
from .model import Encoder_overall
from .preprocess import adjacent_matrix_preprocessing
from .augment import augment_adata


class Train_SpaOmicsVAE:
    def __init__(self, 
        data,
        datatype='SPOTS',
        device=torch.device('cpu'),
        random_seed=2022,
        learning_rate=0.0001,
        weight_decay=0.00,
        epochs=600, 
        dim_input=3000,
        dim_output=64,
        weight_factors=[1, 5, 1, 1]
        ):
        '''
        Parameters
        ----------
        data : dict
            Dictionary containing spatial multi-omics data
        datatype : str, optional
            Input data type. Currently supports 'SPOTS', 'Stereo-CITE-seq', and 'Spatial-ATAC-RNA-seq'
            Future releases will support additional data types
            Default is 'SPOTS'
        device : torch.device, optional
            Computation device (GPU/CPU)
            Default is 'cpu'
        random_seed : int, optional
            Random seed for model initialization reproducibility
            Default is 2022
        learning_rate : float, optional
            Learning rate for spatial transcriptomics representation learning
            Default is 0.001
        weight_decay : float, optional
            Weight decay parameter for controlling weight parameter influence
            Default is 0.00
        epochs : int, optional
            Number of training epochs
            Default is 1500
        dim_input : int, optional
            Input feature dimension
            Default is 3000
        dim_output : int, optional
            Output representation dimension
            Default is 64
        weight_factors : list, optional
            List of weight factors to balance different omics data influences during training
            Default is [1, 5, 1, 1]
        
        Returns
        -------
        self.emb_combined : The learned combined representation
        '''
        self.data = data.copy()
        self.datatype = datatype
        self.device = device
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.weight_factors = weight_factors
        
        # Initialize adjacency matrices
        self.adata_omics1 = self.data['adata_omics1']
        self.adata_omics2 = self.data['adata_omics2']
        self.adj = adjacent_matrix_preprocessing(self.adata_omics1, self.adata_omics2)
        self.adj_spatial_omics1 = self.adj['adj_spatial_omics1'].to(self.device)
        self.adj_spatial_omics2 = self.adj['adj_spatial_omics2'].to(self.device)
        self.adj_feature_omics1 = self.adj['adj_feature_omics1'].to(self.device)
        self.adj_feature_omics2 = self.adj['adj_feature_omics2'].to(self.device)
        
        # Initialize features
        self.features_omics1 = torch.FloatTensor(self.adata_omics1.obsm['feat'].copy()).to(self.device)
        self.features_omics2 = torch.FloatTensor(self.adata_omics2.obsm['feat'].copy()).to(self.device)
        
        self.n_cell_omics1 = self.adata_omics1.n_obs
        self.n_cell_omics2 = self.adata_omics2.n_obs
        
        # Set input and output dimensions
        self.dim_input1 = self.features_omics1.shape[1]
        self.dim_input2 = self.features_omics2.shape[1]
        self.dim_output1 = self.dim_output
        self.dim_output2 = self.dim_output
        
        # Configure datatype-specific parameters
        if self.datatype == 'SPOTS':
            self.epochs = 800 
            self.weight_factors = [1, 5, 1, 1]
        elif self.datatype == 'Stereo-CITE-seq':
            self.epochs = 1600 
            self.weight_factors = [1, 10, 10, 10]
        elif self.datatype == '10x':
            self.epochs = 140
            self.weight_factors = [2, 5, 5, 5]
        elif self.datatype == '10x1':
            self.epochs = 200
            self.weight_factors = [2, 5, 5, 5]   
        elif self.datatype == 'Spatial-epigenome-transcriptome': 
            self.epochs = 1200
            self.weight_factors = [2, 4, 4, 4]
        elif self.datatype == 'Human-brain': 
            self.epochs = 1200
            self.weight_factors = [2, 4, 2, 4] 
    
    def vgae_loss(self, reconstructed_omics1, features_omics1, reconstructed_omics2, features_omics2, 
                mu_omics1, log_var_omics1, mu_omics2, log_var_omics2):
        """
        Compute the Variational Graph Autoencoder (VGAE) loss
        Includes reconstruction loss and KL divergence components
        """
        
        # Reconstruction loss calculation
        loss_recon_omics1 = F.mse_loss(reconstructed_omics1, features_omics1)
        loss_recon_omics2 = F.mse_loss(reconstructed_omics2, features_omics2)
        
        # KL divergence loss calculation
        if mu_omics1 is not None and log_var_omics1 is not None:
            kl_loss_omics1 = -0.5 * torch.sum(1 + log_var_omics1 - mu_omics1.pow(2) - log_var_omics1.exp())
        else:
            kl_loss_omics1 = torch.tensor(0.0)
        
        if mu_omics2 is not None and log_var_omics2 is not None:
            kl_loss_omics2 = -0.5 * torch.sum(1 + log_var_omics2 - mu_omics2.pow(2) - log_var_omics2.exp())
        else:
            kl_loss_omics2 = torch.tensor(0.0)
        
        # Compute total loss
        total_loss = loss_recon_omics1 + loss_recon_omics2 + kl_loss_omics1 + kl_loss_omics2
        
        return total_loss, loss_recon_omics1, loss_recon_omics2, kl_loss_omics1, kl_loss_omics2

    def train(self):
        # Initialize model and optimizer
        self.model = Encoder_overall(self.dim_input1, self.dim_output1, self.dim_input2, self.dim_output2).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, 
                                        weight_decay=self.weight_decay)
        self.model.train()
        
        # Training loop
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            results = self.model(self.features_omics1, self.features_omics2, 
                               self.adj_spatial_omics1, self.adj_feature_omics1, 
                               self.adj_spatial_omics2, self.adj_feature_omics2)
            
            # Extract reconstructed data and latent variables
            reconstructed_omics1 = results['emb_recon_omics1']
            reconstructed_omics2 = results['emb_recon_omics2']

            # Check for VAE components
            mu_omics1 = results.get('mu_omics1', None)
            log_var_omics1 = results.get('log_var_omics1', None)
            mu_omics2 = results.get('mu_omics2', None)
            log_var_omics2 = results.get('log_var_omics2', None)

            # Calculate losses
            total_loss, loss_recon_omics1, loss_recon_omics2, kl_loss_omics1, kl_loss_omics2 = self.vgae_loss(
                reconstructed_omics1, self.features_omics1, reconstructed_omics2, self.features_omics2,
                mu_omics1, log_var_omics1, mu_omics2, log_var_omics2)

            # Apply weight factors
            loss = (self.weight_factors[0] * loss_recon_omics1 + 
                   self.weight_factors[1] * loss_recon_omics2 + 
                   self.weight_factors[2] * kl_loss_omics1 + 
                   self.weight_factors[3] * kl_loss_omics2)
            
            # Optimization step
            self.optimizer.zero_grad()
            loss.backward() 
            self.optimizer.step()
        
        print("Model training completed!\n")    
        
        # Generate final embeddings
        with torch.no_grad():
            self.model.eval()
            results = self.model(self.features_omics1, self.features_omics2, 
                               self.adj_spatial_omics1, self.adj_feature_omics1, 
                               self.adj_spatial_omics2, self.adj_feature_omics2)

        # Normalize embeddings
        emb_omics1 = F.normalize(results['emb_latent_omics1'], p=2, eps=1e-12, dim=1)  
        emb_omics2 = F.normalize(results['emb_latent_omics2'], p=2, eps=1e-12, dim=1)
        emb_combined = F.normalize(results['emb_latent_combined'], p=2, eps=1e-12, dim=1)
        
        # Prepare output dictionary
        output = {
            'emb_latent_omics1': emb_omics1.detach().cpu().numpy(),
            'emb_latent_omics2': emb_omics2.detach().cpu().numpy(),
            'SpaOmicsVAE': emb_combined.detach().cpu().numpy(),
            'loss_recon_omics1': loss_recon_omics1.item(),
            'loss_recon_omics2': loss_recon_omics2.item(),
            'kl_loss_omics1': kl_loss_omics1.item(),
            'kl_loss_omics2': kl_loss_omics2.item(),
            'total_loss': total_loss.item()
        }
        
        return output


def get_augment(
        adata,
        adjacent_weight=0.3,
        neighbour_k=4,
        spatial_k=30,
        n_components=100,
        md_dist_type="cosine",
        gb_dist_type="correlation",
        use_morphological=False,
        use_data="raw",
        spatial_type="KDTree",
        random_seed=2024,
        use_ATAC=False
        ):
    """
    Augment the input anndata object with additional features and spatial information
    
    Parameters
    ----------
    adata : AnnData
        Input annotation data object
    adjacent_weight : float, optional
        Weight for adjacent matrix calculation
    neighbour_k : int, optional
        Number of neighbors for graph construction
    spatial_k : int, optional
        Number of spatial neighbors
    n_components : int, optional
        Number of components for dimension reduction
    md_dist_type : str, optional
        Distance metric type for molecular data
    gb_dist_type : str, optional
        Distance metric type for graph-based analysis
    use_morphological : bool, optional
        Whether to use morphological features
    use_data : str, optional
        Type of data to use ('raw' or processed)
    spatial_type : str, optional
        Type of spatial analysis method
    random_seed : int, optional
        Random seed for reproducibility
    use_ATAC : bool, optional
        Whether to use ATAC-seq data
        
    Returns
    -------
    adata : AnnData
        Augmented annotation data object
    """
    adata = augment_adata(adata, 
            md_dist_type=md_dist_type,
            gb_dist_type=gb_dist_type,
            n_components=n_components,
            use_morphological=use_morphological,
            use_data=use_data,
            neighbour_k=neighbour_k,
            adjacent_weight=adjacent_weight,
            spatial_k=spatial_k,
            spatial_type=spatial_type,
            random_seed=random_seed,
            use_ATAC=use_ATAC
            )
    print("Data augmentation completed!")
    return adata