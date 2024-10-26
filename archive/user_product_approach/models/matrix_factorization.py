import torch.nn as nn

class MatrixFactorizationLogitModel(nn.Module):
    def __init__(self, n_users, n_products, n_factors=50):
        super(MatrixFactorizationLogitModel, self).__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.product_factors = nn.Embedding(n_products, n_factors)
        self.logit_embedding = nn.Linear(n_factors, 5)

    def forward(self, user, product):
        user_embedding = self.user_factors(user)
        product_embedding = self.product_factors(product)
        interaction = user_embedding * product_embedding  # Element-wise product
        logits = self.logit_embedding(interaction)  # Pass through secondary embedding to get logits
        return logits
