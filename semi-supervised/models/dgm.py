import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from .vae import VariationalAutoencoder
from .vae import Encoder, Decoder, LadderEncoder, LadderDecoder


class Classifier(nn.Module):
    def __init__(self, dims):
        """
        Single hidden layer classifier.

        dims: [x_dim, h_dim, y_dim]
        """
        super().__init__()
        x_dim, h_dim, y_dim = dims
        self.dense = nn.Linear(x_dim, h_dim)
        self.labels = nn.Linear(h_dim, y_dim)
        self.batch_norm1 = nn.BatchNorm1d(h_dim)

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.dense(x)))
        x = self.labels(x)
        return torch.sigmoid(x)


class DeepGenerativeModel(VariationalAutoencoder):
    def __init__(self, dims):
        """
        M2 model ('Semi-Supervised Learning with Deep Generative Models', Kingma 2014).

        dims: [x_dim, y_dim, z_dim, h_dim_list]
        """
        x_dim, self.y_dim, z_dim, h_dim = dims
        super().__init__([x_dim, z_dim, h_dim])

        self.encoder = Encoder([x_dim + self.y_dim, h_dim, z_dim])
        self.decoder = Decoder([z_dim + self.y_dim, list(reversed(h_dim)), x_dim])
        self.classifier = Classifier([x_dim, h_dim[0], self.y_dim])

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, y):
        # q(z | x, y)
        z, z_mu, z_log_var = self.encoder(torch.cat([x, y], dim=1))
        self.kl_divergence = self._kld(z, (z_mu, z_log_var))

        # p(x | z, y)
        x_mu = self.decoder(torch.cat([z, y], dim=1))
        return x_mu

    def classify(self, x):
        return self.classifier(x)

    def sample(self, z, y):
        """
        Sample x ~ p(x | z, y)
        """
        y = y.float()
        x = self.decoder(torch.cat([z, y], dim=1))
        return x


class StackedDeepGenerativeModel(DeepGenerativeModel):
    def __init__(self, dims, features):
        """
        M1+M2 stacked model (Kingma 2014).

        dims: [x_dim, y_dim, z_dim, h_dim_list]
        features: pretrained VariationalAutoencoder (M1) on same data.
        """
        x_dim, y_dim, z_dim, h_dim = dims
        super().__init__([features.z_dim, y_dim, z_dim, h_dim])

        # adjust final reconstruction to original x_dim
        in_features = self.decoder.reconstruction.in_features
        self.decoder.reconstruction = nn.Linear(in_features, x_dim)

        # freeze M1 features
        self.features = features
        self.features.train(False)
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x, y):
        # encode via M1 to a latent representation, then use M2
        x_sample, _, _ = self.features.encoder(x)
        return super().forward(x_sample, y)

    def classify(self, x):
        # often mu can be used; this keeps behavior consistent with your fork
        _, x_mu, _ = self.features.encoder(x)
        return self.classifier(x_mu)


class AuxiliaryDeepGenerativeModel(DeepGenerativeModel):
    def __init__(self, dims):
        """
        ADGM (Maaløe 2016): adds auxiliary latent variable a.

        dims: [x_dim, y_dim, z_dim, a_dim, h_dim_list]
        """
        x_dim, y_dim, z_dim, a_dim, h_dim = dims
        super().__init__([x_dim, y_dim, z_dim, h_dim])

        self.aux_encoder = Encoder([x_dim, h_dim, a_dim])                           # q(a | x)
        self.aux_decoder = Encoder([x_dim + z_dim + y_dim, list(reversed(h_dim)), a_dim])  # p(a | x, z, y)

        self.classifier = Classifier([x_dim + a_dim, h_dim[0], y_dim])              # q(y | x, a)

        self.encoder = Encoder([a_dim + y_dim + x_dim, h_dim, z_dim])               # q(z | x, y, a)
        self.decoder = Decoder([y_dim + z_dim, list(reversed(h_dim)), x_dim])       # p(x | z, y)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def classify(self, x):
        # q(a | x) then q(y | x, a)
        a, a_mu, a_log_var = self.aux_encoder(x)
        labels = self.classifier(torch.cat([x, a], dim=1))
        return labels

    def forward(self, x, y):
        """
        Forward pass computing q(a|x), q(z|x,y,a), p(a|x,y,z), p(x|z,y)
        and accumulates KL terms.
        """
        # q(a | x)
        q_a, q_a_mu, q_a_log_var = self.aux_encoder(x)

        # q(z | x, y, a)
        z, z_mu, z_log_var = self.encoder(torch.cat([x, y, q_a], dim=1))

        # p(x | z, y)
        x_mu = self.decoder(torch.cat([z, y], dim=1))

        # p(a | x, y, z)
        p_a, p_a_mu, p_a_log_var = self.aux_decoder(torch.cat([x, y, z], dim=1))

        a_kl = self._kld(q_a, (q_a_mu, q_a_log_var), (p_a_mu, p_a_log_var))
        z_kl = self._kld(z, (z_mu, z_log_var))

        self.kl_divergence = a_kl + z_kl
        return x_mu


class LadderDeepGenerativeModel(DeepGenerativeModel):
    def __init__(self, dims):
        """
        Ladder DGM: hierarchical z’s.

        dims: [x_dim, y_dim, [z_dims_top_to_bottom], [h_dims]]
               len(z_dims) == len(h_dims)
        """
        x_dim, y_dim, z_dim, h_dim = dims
        super().__init__([x_dim, y_dim, z_dim[0], h_dim])

        neurons = [x_dim, *h_dim]

        # encoders bottom-up; last one takes y as well
        encoder_layers = [LadderEncoder([neurons[i - 1], neurons[i], z_dim[i - 1]]) for i in range(1, len(neurons))]
        last = encoder_layers[-1]
        encoder_layers[-1] = LadderEncoder([last.in_features + y_dim, last.out_features, last.z_dim])

        # decoders top-down (reverse order)
        decoder_layers = [LadderDecoder([z_dim[i - 1], h_dim[i - 1], z_dim[i]]) for i in range(1, len(h_dim))][::-1]

        self.classifier = Classifier([x_dim, h_dim[0], y_dim])

        self.encoder = nn.ModuleList(encoder_layers)
        self.decoder = nn.ModuleList(decoder_layers)
        self.reconstruction = Decoder([z_dim[0] + y_dim, h_dim, x_dim])

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, y):
        # encode ladder
        latents = []
        for i, enc in enumerate(self.encoder):
            if i == len(self.encoder) - 1:
                x, (z, mu, log_var) = enc(torch.cat([x, y], dim=1))
            else:
                x, (z, mu, log_var) = enc(x)
            latents.append((mu, log_var))

        latents = list(reversed(latents))

        # top-level KL
        self.kl_divergence = 0.0
        top_mu, top_log_var = latents[0]
        self.kl_divergence += self._kld(z, (top_mu, top_log_var))

        # descend ladder with merges + KLs
        cur_z = z
        for i, dec in enumerate(self.decoder, start=1):
            l_mu, l_log_var = latents[i]
            cur_z, kl_bits = dec(cur_z, l_mu, l_log_var)
            self.kl_divergence += self._kld(*kl_bits)

        x_mu = self.reconstruction(torch.cat([cur_z, y], dim=1))
        return x_mu

    def sample(self, z, y):
        cur = z
        for dec in self.decoder:
            cur = dec(cur)
        return self.reconstruction(torch.cat([cur, y], dim=1))
