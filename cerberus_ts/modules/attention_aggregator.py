import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionAggregator(nn.Module):
    def __init__(self, channels, seq_length, feature_length, d_neck):
        super(AttentionAggregator, self).__init__()
        self.seq_length = seq_length
        self.attention_weights = nn.Linear(channels * feature_length, seq_length)
        self.fc = nn.Linear(channels * feature_length, d_neck)

    def forward(self, x):
        batches, channels, seq_length, feature_length = x.size() # [batches, channels, seq_length, feature_length]

        # Flatten channels and feature_length and transpose
        x = x.view(batches, seq_length, channels * feature_length) # [batches, seq_length, channels * feature_length]

        # Compute attention weights
        attention = F.softmax(self.attention_weights(x), dim=-1) # [batches, seq_length, seq_length]

        # Apply attention - weighted sum across seq_length
        attended = torch.bmm(attention, x).sum(dim=1) # [batches, channels * feature_length]

        # Reduce
        out = self.fc(attended) #[batches, d_neck]

        return out

if __name__=="__main__":
    # Example usage
    batches = 10
    channels = 3
    seq_length = 32
    feature_length = 32
    d_neck = 24

    model = AttentionAggregator(channels, seq_length, feature_length, d_neck)

    input_tensor = torch.randn(batches, channels, seq_length, feature_length)  # Input dimensions: [batches, channels, seq_length, feature_length]
    output_tensor = model(input_tensor)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")
