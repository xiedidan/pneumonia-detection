import torch

def check_bbox(dets):
    results = dets

    # filter out bbox out of borders
    for i in range(4):
        if results.dim() == 0 or results.dim() == 1:
            return torch.tensor(0.)

        idx = i + 1

        mask_min = results[:, idx].gt(0.).expand(5, results.size(0)).t()
        mask_max = results[:, idx].lt(1.).expand(5, results.size(0)).t()
        mask = torch.min(mask_min, mask_max).to(dtype=torch.uint8) # and op

        results = torch.masked_select(results, mask).view(-1, 5)

    if results.dim() == 0 or results.dim() == 1:
        return torch.tensor(0.)

    # filter out bbox larger than 0.4 * total area
    mask = []
    for det in results:
        area = (det[3] - det[1]) * (det[4] - det[2])

        if area > 0.4:
            mask.append(0)
        else:
            mask.append(1)
    mask = torch.tensor(mask).expand(5, results.size(0)).t().to(dtype=torch.uint8)

    results = torch.masked_select(results, mask).view(-1, 5)

    if results.dim() == 0 or results.dim() == 1:
        return torch.tensor(0.)

    return results
