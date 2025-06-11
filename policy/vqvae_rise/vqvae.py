import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import jit
from policy.vqvae_rise.vqvae_utils import *
import einops
from policy.vqvae_rise.vector_quantize_pytorch.residual_vq import ResidualVQ
from torch.utils.data import DataLoader, TensorDataset

class EncoderMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim=16,
        hidden_dim=128,
        layer_num=1,
        last_activation=None,
    ):
        super(EncoderMLP, self).__init__()
        layers = []

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(layer_num):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

        if last_activation is not None:
            self.last_layer = last_activation
        else:
            self.last_layer = None
        self.apply(weights_init_encoder)

    def forward(self, x):
        h = self.encoder(x)
        state = self.fc(h)
        if self.last_layer:
            state = self.last_layer(state)
        return state


class VqVae:
    def __init__(
        self,
        obs_dim=60,
        input_dim_h=10,  # length of action chunk
        input_dim_w=9,  # action dim
        n_latent_dims=512,
        vqvae_n_embed=32,
        vqvae_groups=4,
        eval=True,
        device="cuda",
        load_dir=None,
        encoder_loss_multiplier=1.0,
        act_scale=1.0,
    ):
        self.n_latent_dims = n_latent_dims
        self.input_dim_h = input_dim_h
        self.input_dim_w = input_dim_w
        self.rep_dim = self.n_latent_dims
        self.vqvae_n_embed = vqvae_n_embed
        self.vqvae_lr = 1e-3
        self.vqvae_groups = vqvae_groups
        self.device = device
        self.encoder_loss_multiplier = encoder_loss_multiplier
        self.act_scale = act_scale

        discrete_cfg = {"groups": self.vqvae_groups, "n_embed": self.vqvae_n_embed}

        self.vq_layer = ResidualVQ(
            dim=self.n_latent_dims,
            num_quantizers=discrete_cfg["groups"],
            codebook_size=self.vqvae_n_embed,
        ).to(self.device)
        self.embedding_dim = self.n_latent_dims

        self.vq_layer.device = device

        if self.input_dim_h == 1:
            self.encoder = EncoderMLP(
                input_dim=input_dim_w, output_dim=n_latent_dims
            ).to(self.device)
            self.decoder = EncoderMLP(
                input_dim=n_latent_dims, output_dim=input_dim_w
            ).to(self.device)
        else:
            self.encoder = EncoderMLP(
                input_dim=input_dim_w * self.input_dim_h, output_dim=n_latent_dims
            ).to(self.device)
            self.decoder = EncoderMLP(
                input_dim=n_latent_dims, output_dim=input_dim_w * self.input_dim_h
            ).to(self.device)

        params = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.vq_layer.parameters())
        )
        self.vqvae_optimizer = torch.optim.Adam(
            params, lr=self.vqvae_lr, weight_decay=0.0001
        )

        if load_dir is not None:
            try:
                state_dict = torch.load(load_dir)
            except RuntimeError:
                state_dict = torch.load(load_dir, map_location=torch.device("cpu"))
            self.load_state_dict(state_dict)

        if eval:
            self.vq_layer.eval()
        else:
            self.vq_layer.train()

    def draw_logits_forward(self, encoding_logits):
        z_embed = self.vq_layer.draw_logits_forward(encoding_logits)
        return z_embed

    def draw_code_forward(self, encoding_indices):
        with torch.no_grad():
            z_embed = self.vq_layer.get_codes_from_indices(encoding_indices)
            z_embed = z_embed.sum(dim=0)
        return z_embed

    def get_action_from_latent(self, latent):
        output = self.decoder(latent) * self.act_scale
        if self.input_dim_h == 1:
            return einops.rearrange(output, "N (T A) -> N T A", A=self.input_dim_w)
        else:
            return einops.rearrange(output, "N (T A) -> N T A", A=self.input_dim_w)

    def preprocess(self, state):
        if not torch.is_tensor(state):
            state = get_tensor(state, self.device)
        if self.input_dim_h == 1:
            state = state.squeeze(-2)  # state.squeeze(-1)
        else:
            state = einops.rearrange(state, "N T A -> N (T A)")
        return state.to(self.device)

    def get_code(self, state, required_recon=False):
        state = state / self.act_scale
        state = self.preprocess(state)
        with torch.no_grad():
            state_rep = self.encoder(state)
            state_rep_shape = state_rep.shape[:-1]
            state_rep_flat = state_rep.view(state_rep.size(0), -1, state_rep.size(1))
            state_rep_flat, vq_code, vq_loss_state = self.vq_layer(state_rep_flat)
            state_vq = state_rep_flat.view(*state_rep_shape, -1)
            vq_code = vq_code.view(*state_rep_shape, -1)
            vq_loss_state = torch.sum(vq_loss_state)
            if required_recon:
                recon_state = self.decoder(state_vq) * self.act_scale
                recon_state_ae = self.decoder(state_rep) * self.act_scale
                if self.input_dim_h == 1:
                    return state_vq, vq_code, recon_state, recon_state_ae
                else:
                    return (
                        state_vq,
                        vq_code,
                        torch.swapaxes(recon_state, -2, -1),
                        torch.swapaxes(recon_state_ae, -2, -1),
                    )
            else:
                # econ_from_code = self.draw_code_forward(vq_code)
                return state_vq, vq_code

    def vqvae_update(self, state):
        state = state / self.act_scale # state [B, l, dim]
        state = self.preprocess(state) # [B, l*dim]
        state_rep = self.encoder(state) # [B, n_latent_dims]
        state_rep_shape = state_rep.shape[:-1]
        state_rep_flat = state_rep.view(state_rep.size(0), -1, state_rep.size(1)) # [B, 1, n_latent_dims]
        state_rep_flat, vq_code, vq_loss_state = self.vq_layer(state_rep_flat)
        # state_rep_flat: [B, 1, n_latent_dims]
        # vq_code [B,1,vq_code_dim] e.g. vq_code_dim =2: 2 int encode a token
        # vq_loss_state [1, vq_code_dim]
        state_vq = state_rep_flat.view(*state_rep_shape, -1)
        # state_vq: [B, l*dim]
        vq_code = vq_code.view(*state_rep_shape, -1)
        # vq_code: [B, vq_code_dim]
        vq_loss_state = torch.sum(vq_loss_state)

        dec_out = self.decoder(state_vq)
        # dec_out [B, l*dim]
        encoder_loss = (state - dec_out).abs().mean()
        # encoder_loss is the reconstruction loss
        rep_loss = encoder_loss * self.encoder_loss_multiplier + (vq_loss_state * 5) # Weight and Sum the two losses

        # Optimize the critic
        self.vqvae_optimizer.zero_grad()
        rep_loss.backward()
        self.vqvae_optimizer.step()
        vqvae_recon_loss = torch.nn.MSELoss()(state, dec_out)
        return (
            encoder_loss.clone().detach(),
            vq_loss_state.clone().detach(),
            vq_code,
            vqvae_recon_loss.item(),
        )

    def state_dict(self):
        return {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "optimizer": self.vqvae_optimizer.state_dict(),
            "vq_embedding": self.vq_layer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict["encoder"])
        self.decoder.load_state_dict(state_dict["decoder"])
        self.vqvae_optimizer.load_state_dict(state_dict["optimizer"])
        self.vq_layer.load_state_dict(state_dict["vq_embedding"])
        self.vq_layer.eval()

def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def generate_dummy_dataset(num_samples=1000, seq_len=2, action_dim=3):
    """
    Create a simple dataset of [N, T, A] shape actions for training.
    """
    data = np.random.uniform(-1, 1, size=(num_samples, seq_len, action_dim)).astype(np.float32)
    return torch.tensor(data)

# if __name__ == '__main__':
#     print("Import Packages Successfully")

#     # Instantiate the model
#     vqvae = VqVae(
#         obs_dim=60,
#         input_dim_h=2,     # Let's use 2 time steps
#         input_dim_w=3,     # Each action is 3D
#         n_latent_dims=16,  # Keep it small
#         vqvae_n_embed=8,
#         vqvae_groups=2,
#         eval=True,
#         device="cuda",      # use CPU to avoid needing CUDA
#     )

#     # Create dummy input data: 2 samples, 2 time steps, 3-dim actions
#     # Shape: (N, T, A)
#     dummy_data = torch.tensor([
#         [[1.0, 0.5, -0.5], [0.1, 0.2, 0.3]],  # Sample 1
#         [[-0.3, 0.0, 1.0], [0.9, -0.1, -0.2]]  # Sample 2
#     ], dtype=torch.float32)

#     print("\nDummy Input:\n", dummy_data)

#     # Forward through encoder + VQ + decoder
#     state_vq, vq_code = vqvae.get_code(dummy_data, required_recon=False)

#     print("\nVQ Code:\n", vq_code)
#     print("\nLatent (quantized state):\n", state_vq)

#     # Decode the latent representation
#     recon_action = vqvae.get_action_from_latent(state_vq)
#     print("\nReconstructed Action:\n", recon_action)
def main():
    seed_everything()
    
    # Hyperparameters
    batch_size = 32
    epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    vqvae_model = VqVae(
        obs_dim=60,
        input_dim_h=2,         # Sequence length
        input_dim_w=3,         # Action dimension
        n_latent_dims=16,
        vqvae_n_embed=8,
        vqvae_groups=2,
        eval=False,
        device=device,
    )

    # Dataset
    data = generate_dummy_dataset(num_samples=512, seq_len=2, action_dim=3)
    dataset = TensorDataset(data)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # Training loop
    for epoch in range(epochs):
        total_loss, total_recon, total_vq = 0, 0, 0
        for batch in train_loader:
            act = batch[0].to(device)
            encoder_loss, vq_loss_state, vq_code, recon_loss = vqvae_model.vqvae_update(act)

            total_loss += encoder_loss.item() + vq_loss_state.item()
            total_recon += recon_loss
            total_vq += vq_loss_state.item()

        print(f"[Epoch {epoch+1}] Total Loss: {total_loss:.4f}, Recon: {total_recon:.4f}, VQ: {total_vq:.4f}")

    # Optional: Save model
    torch.save(vqvae_model.state_dict(), "trained_vqvae.pt")

if __name__ == "__main__":
    main()