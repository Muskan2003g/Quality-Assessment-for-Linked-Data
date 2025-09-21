import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from layers import GaussianSample, GaussianMerge, GumbelSoftmax
from inference import log_gaussian, log_standard_gaussian


class Perceptron(nn.Module):
    def __init__(self, dims, activation_fn=F.relu, output_activation=None):
        super().__init__()
        self.dims = dims
        self.activation_fn = activation_fn
        self.output_activation = output_activation
        self.layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(dims, dims[1:])])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers) - 1 and self.output_activation is not None:
                x = self.output_activation(x)
            else:
                x = self.activation_fn(x)
        return x


class Encoder(nn.Module):
    def __init__(self, dims, sample_layer=GaussianSample):
        """
        dims: [input_dim, [hidden_dims], latent_dim]
        """
        super().__init__()
        x_dim, h_dim, z_dim = dims
        neurons = [x_dim, *h_dim]
        self.hidden = nn.ModuleList([nn.Linear(neurons[i - 1], neurons[i]) for i in range(1, len(neurons))])
        self.batch_norm = nn.ModuleList([nn.BatchNorm1d(neurons[i]) for i in range(1, len(neurons))])
        self.sample = sample_layer(h_dim[-1], z_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        for index, layer in enumerate(self.hidden):
            x = F.relu(self.batch_norm[index](layer(x)))
        return self.sample(x)  # returns (z, mu, log_var)


class Decoder(nn.Module):
    def __init__(self, dims):
        """
        dims: [latent_dim, [hidden_dims], input_dim]
        """
        super().__init__()
        z_dim, h_dim, x_dim = dims
        neurons = [z_dim, *h_dim]
        self.hidden = nn.ModuleList([nn.Linear(neurons[i - 1], neurons[i]) for i in range(1, len(neurons))])
        self.batch_norm = nn.ModuleList([nn.BatchNorm1d(neurons[i]) for i in range(1, len(neurons))])
        self.reconstruction = nn.Linear(h_dim[-1], x_dim)
        self.output_activation = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        for index, layer in enumerate(self.hidden):
            x = F.relu(self.batch_norm[index](layer(x)))
        return self.output_activation(self.reconstruction(x))


class VariationalAutoencoder(nn.Module):
    def __init__(self, dims):
        """
        dims: [x_dim, z_dim, [hidden_dims]]
        """
        super().__init__()
        x_dim, z_dim, h_dim = dims
        self.z_dim = z_dim
        self.flow = None

        self.encoder = Encoder([x_dim, h_dim, z_dim])
        self.decoder = Decoder([z_dim, list(reversed(h_dim)), x_dim])
        self.kl_divergence = 0.0

    def _kld(self, z, q_param, p_param=None):
        (mu, log_var) = q_param

        if self.flow is not None:
            f_z, log_det_z = self.flow(z)
            qz = log_gaussian(z, mu, log_var) - sum(log_det_z)
            z = f_z
        else:
            qz = log_gaussian(z, mu, log_var)

        if p_param is None:
            pz = log_standard_gaussian(z)
        else:
            (p_mu, p_log_var) = p_param
            pz = log_gaussian(z, p_mu, p_log_var)

        kl = qz - pz
        return kl

    def add_flow(self, flow):
        self.flow = flow

    def forward(self, x, y=None):
        z, z_mu, z_log_var = self.encoder(x)
        self.kl_divergence = self._kld(z, (z_mu, z_log_var))
        x_mu = self.decoder(z)
        return x_mu

    def sample(self, z):
        return self.decoder(z)


class GumbelAutoencoder(nn.Module):
    def __init__(self, dims, n_samples=100):
        """
        dims: [x_dim, z_dim(=classes), [hidden_dims]]
        """
        super().__init__()
        x_dim, z_dim, h_dim = dims
        self.z_dim = z_dim
        self.n_samples = n_samples

        self.encoder = Perceptron([x_dim, *h_dim])
        self.sampler = GumbelSoftmax(h_dim[-1], z_dim, n_samples)
        self.decoder = Perceptron([z_dim, *reversed(h_dim), x_dim], output_activation=torch.sigmoid)

        self.kl_divergence = 0.0

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _kld(self, qz):
        # qz shape expected: [B * n_samples, z_dim] (from sampler); handle general case too.
        eps = 1e-8
        k = torch.tensor(self.z_dim, dtype=qz.dtype, device=qz.device)
        kl = qz * (torch.log(qz + eps) - torch.log(1.0 / k))
        kl = kl.view(-1, self.n_samples, self.z_dim)
        return torch.sum(torch.sum(kl, dim=1), dim=1)

    def forward(self, x, y=None, tau=1.0):
        x = self.encoder(x)
        sample, qz = self.sampler(x, tau)
        self.kl_divergence = self._kld(qz)
        x_mu = self.decoder(sample)
        return x_mu

    def sample(self, z):
        return self.decoder(z)


class LadderEncoder(nn.Module):
    def __init__(self, dims):
        """
        dims: [input_dim, hidden_dim, latent_dim]
        """
        super().__init__()
        x_dim, h_dim, z_dim = dims
        self.linear = nn.Linear(x_dim, h_dim)
        self.batchnorm = nn.BatchNorm1d(h_dim)
        self.sample = GaussianSample(h_dim, z_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.linear(x)
        x = F.leaky_relu(self.batchnorm(x), 0.1)
        return x, self.sample(x)  # (z, mu, log_var)


class LadderDecoder(nn.Module):
    def __init__(self, dims):
        """
        dims: [latent_dim_in, hidden_dim, latent_dim_out]
        """
        super().__init__()
        z_in, h_dim, z_out = dims

        self.linear1 = nn.Linear(z_in, h_dim)
        self.batchnorm1 = nn.BatchNorm1d(h_dim)
        self.merge = GaussianMerge(h_dim, z_out)

        self.linear2 = nn.Linear(z_in, h_dim)
        self.batchnorm2 = nn.BatchNorm1d(h_dim)
        self.sample = GaussianSample(h_dim, z_out)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, l_mu=None, l_log_var=None):
        if l_mu is not None:
            z = self.linear1(x)
            z = F.leaky_relu(self.batchnorm1(z), 0.1)
            q_z, q_mu, q_log_var = self.merge(z, l_mu, l_log_var)

        z = self.linear2(x)
        z = F.leaky_relu(self.batchnorm2(z), 0.1)
        z, p_mu, p_log_var = self.sample(z)

        if l_mu is None:
            return z

        return z, (q_z, (q_mu, q_log_var), (p_mu, p_log_var))


class LadderVariationalAutoencoder(VariationalAutoencoder):
    def __init__(self, dims):
        """
        dims: [x_dim, [z_dims_top_to_bottom], [hidden_dims]]
        """
        x_dim, z_dim, h_dim = dims
        # top layer VAE (use topmost z)
        super().__init__([x_dim, z_dim[0], h_dim])

        neurons = [x_dim, *h_dim]
        # encoders go bottom-up: x_dim -> h_dim[i] -> z_dim[i]
        self.encoder = nn.ModuleList([
            LadderEncoder([neurons[i - 1], neurons[i], z_dim[i - 1]])
            for i in range(1, len(neurons))
        ])
        # decoders go top-down between latent layers; reverse for decode order
        self.decoder = nn.ModuleList([
            LadderDecoder([z_dim[i - 1], h_dim[i - 1], z_dim[i]])
            for i in range(1, len(h_dim))
        ][::-1])
        # final reconstruction from topmost z to x
        self.reconstruction = Decoder([z_dim[0], h_dim, x_dim])

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # encode up the ladder
        latents = []
        for enc in self.encoder:
            x, (z, mu, log_var) = enc(x)
            latents.append((mu, log_var))
        latents = list(reversed(latents))  # top to bottom

        # KL at the top (prior)
        self.kl_divergence = 0.0
        top_mu, top_log_var = latents[0]
        self.kl_divergence += self._kld(z, (top_mu, top_log_var))

        # walk down the ladder, merging information
        cur_z = z
        for i, dec in enumerate(self.decoder, start=1):
            l_mu, l_log_var = latents[i]
            cur_z, kl_bits = dec(cur_z, l_mu, l_log_var)  # (q_z, (q_mu, q_log_var), (p_mu, p_log_var))
            self.kl_divergence += self._kld(*kl_bits)

        x_mu = self.reconstruction(cur_z)
        return x_mu

    def sample(self, z):
        cur = z
        for dec in self.decoder:
            cur = dec(cur)
        return self.reconstruction(cur)
