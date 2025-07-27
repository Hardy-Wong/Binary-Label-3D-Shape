from typing import List, Optional

import torch
from torch import Tensor, nn

from common.consts import NORMALIZE_RATIO


class AbsolutePositionMeshLinearVAE(nn.Module):
    """Only Positional feature model compared to original MeshVAE.

    The input: mesh node positions and mean log rotation, stretch tensor * (num of nodes) vector
    The output: mesh node positions.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: Optional[List[int]],
        output_dim: int,
        sigma_max: float = 2.0,
        scale: float = 1e3,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sigma_max = sigma_max
        self.scale = scale

        modules = []
        if hidden_dims is None:
            hidden_dims = [1024, 256]
        if len(hidden_dims) < 1:
            raise ValueError("hidden_dims must have more than 1 items.")

        # Build Encoder
        tmp = input_dim
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_features=tmp, out_features=h_dim, dtype=torch.float),
                    nn.BatchNorm1d(h_dim, dtype=torch.float),
                    nn.LeakyReLU(),
                )
            )
            tmp = h_dim
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Sequential(
            nn.Linear(hidden_dims[-1], latent_dim), nn.Sigmoid()
        )

        # Build Decoder
        modules = []

        hidden_dims.reverse()
        tmp = latent_dim
        for hidden_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(
                        in_features=tmp, out_features=hidden_dim, dtype=torch.float
                    ),
                    nn.BatchNorm1d(hidden_dim, dtype=torch.float),
                    nn.LeakyReLU(),
                )
            )
            tmp = hidden_dim

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.Linear(
                in_features=hidden_dims[-1], out_features=output_dim
            ),  # output feature is the all nodes position.
            nn.Tanh()
            # the output is not in [-1, 1]
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        sigma = self.fc_var(result) * self.sigma_max

        return [mu, sigma]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder(z)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, sigma: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param sigma: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        eps = torch.randn_like(sigma)
        return eps * sigma + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, sigma = self.encode(input)
        z = self.reparameterize(mu, sigma)
        return [self.decode(z), input, mu, sigma]

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        sigma = args[3]

        recon_weight = kwargs["M_N"]
        # Account for the minibatch samples from the dataset
        # select only positions.
        batch_size = len(input)
        recon_errors = (
            torch.sqrt(
                torch.sum(
                    torch.square(
                        input.reshape((batch_size, -1, 3))
                        - recons.reshape((batch_size, -1, 3))
                    ),
                    dim=2,
                )
            )
            * NORMALIZE_RATIO
        )
        recons_loss = torch.mean(recon_errors)
        # The size of input and the size of output is different.

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 - sigma**2 - mu**2 + (sigma**2).log(), dim=1),
            dim=0,
        )

        loss = recon_weight * recons_loss + kld_loss
        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss.detach(),
            "KLD": kld_loss.detach(),
        }

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
