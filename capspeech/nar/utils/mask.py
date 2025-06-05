import torch


def create_tts_mask(seq_len, max_seq_len, mask_range):

    bs = seq_len.size(0)
    device = seq_len.device

    # 1. Sample random fractional lengths for each sequence
    frac_lengths = torch.zeros(bs, device=device).uniform_(*mask_range)

    # 2. Convert fractional lengths to integer lengths
    lengths = (frac_lengths * seq_len).long()

    # 3. Compute valid start indices based on sequence length
    max_start = seq_len - lengths

    # 4. Sample random start positions (clamped at 0)
    rand = torch.rand(bs, device=device)
    start = (max_start * rand).long().clamp(min=0)
    end = start + lengths

    # 5. Build the final boolean mask
    # max_seq_len = seq_len.max().item()
    seq = torch.arange(max_seq_len, device=device).long()

    start_mask = seq[None, :] >= start[:, None]
    end_mask = seq[None, :] <  end[:, None]
    mask = start_mask & end_mask

    return mask


if __name__ == "__main__":
    # Example: 3 sequences of lengths [5, 7, 6]
    lengths = torch.tensor([5, 7, 6])
    mask_range = (0.7, 1.0)  # Sample fractional lengths between 30% and 70% of each seq

    mask = create_tts_mask(lengths, mask_range)
    print("Mask shape:", mask.shape)  # Should be [3, 7], since max_seq_len is 7
    print(mask)