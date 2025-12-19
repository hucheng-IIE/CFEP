import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import pickle

class Generator(nn.Module):
    def __init__(self, emb_dim, lr):
        super(Generator, self).__init__()

        self.gen_w = nn.Parameter(torch.empty(emb_dim, emb_dim))
        nn.init.xavier_uniform_(self.gen_w)

        self.optimizer_G = torch.optim.Adam(self.parameters(), lr=lr)
        self.generator_loss = nn.BCELoss()
         
    def forward(self, rel_embedding):  
        #(barch_size,num_rels,dim)
        noise_embedding = np.random.normal(loc=0.0, scale=1.0, size=rel_embedding.shape)
        noise_embedding = torch.tensor(noise_embedding, dtype=torch.float32).cuda()
        # with open('/data3/hucheng/hucheng/IJCAI_2025/src/embedding/embedding_noise'+'_Sead'+'.pkl', 'wb') as f:
        #     pickle.dump(noise_embedding, f)
        fake_rel_embedding = rel_embedding + noise_embedding
        #fake_rel_embedding = F.leaky_relu(torch.matmul(input, self.gen_w))

        return fake_rel_embedding

    def update(self, discriminator, fake_embeddings):
        self.optimizer_G.zero_grad()
        real_labels = torch.ones(fake_embeddings.size(0), fake_embeddings.size(1), 1).cuda()
        fake_validity = discriminator(fake_embeddings)
        g_loss = self.generator_loss(fake_validity, real_labels)
        g_loss.backward(retain_graph=True)
        self.optimizer_G.step()
        return g_loss.item()