import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _nms(heat, kernel=3):
    """Local-maximum suppression on a heat-map.
    """
    if kernel is None or int(kernel) <= 1:
        return heat
    kernel = int(kernel)
    if kernel % 2 == 0:                 # force odd -> centred window
        kernel += 1
    pad = (kernel - 1) // 2
    # fp32 for the pooling: max_pool2d_with_indices is not guaranteed
    # for bf16 under bf16-mixed precision.
    hmax, hidx = F.max_pool2d(
        heat.float(), (kernel, kernel), stride=1, padding=pad,
        return_indices=True)
    b, c, h, w = heat.shape
    flat = torch.arange(h * w, device=heat.device).view(1, 1, h, w)
    keep = (hidx == flat).to(heat.dtype)
    return heat * keep

def _bev_nms_mask(xy, scores, radius=0.0, boxes=None, iou_thresh=0.0,
                  score_thresh=0.0):
    """Greedy NMS over the Top-K decoded peaks of ONE batch element.

    xy      : (K, 2) BEV *memory* coords (sub-pixel)
    scores  : (K,)
    boxes   : (K, 5) = (cx, cy, length, width, yaw) in BEV CELLS, or None
    radius  : centre-distance gate, in BEV CELLS (0 = off)
    iou_thresh : rotated-BEV-IoU gate (0 = off, needs `boxes`)

    Returns a (K,) bool tensor: True = SUPPRESSED (duplicate).
    Only peaks with score > score_thresh take part, so this costs
    nothing while the heat-map is still cold.
    """
    K = int(scores.shape[0])
    suppressed = np.zeros(K, dtype=bool)
    if K == 0 or (radius <= 0 and iou_thresh <= 0):
        return torch.from_numpy(suppressed).to(scores.device)

    s = scores.detach().float().cpu().numpy().reshape(K)
    p = xy.detach().float().cpu().numpy().reshape(K, 2)

    bx, iou_mat = None, None
    if boxes is not None and iou_thresh > 0:
        try:
            from evaluation.obb_iou import obb_iou_matrix as iou_mat
            bx = boxes.detach().float().cpu().numpy().reshape(K, 5)
        except Exception:               # never break decoding
            bx, iou_mat = None, None

    order = np.argsort(-s)
    order = order[s[order] > score_thresh]
    r2 = float(radius) ** 2

    for pos, i in enumerate(order):
        if suppressed[i]:
            continue                    # a killed peak may not kill others
        rest = order[pos + 1:]
        rest = rest[~suppressed[rest]]  # only LOWER-scoring survivors
        if rest.size == 0:
            break
        kill = np.zeros(rest.shape[0], dtype=bool)
        if radius > 0:
            kill |= ((p[rest] - p[i]) ** 2).sum(1) < r2
        if bx is not None:
            kill |= (iou_mat(bx[i:i + 1], bx[rest], prefilter=True)[0] >= iou_thresh)
        suppressed[rest[kill]] = True

    return torch.from_numpy(suppressed).to(scores.device)

def get_box_from_corners(corners):
    """"
    corners: 4,2
    """
    xmin = torch.min(corners[:, 0], dim=0, keepdim=True).values
    xmax = torch.max(corners[:, 0], dim=0, keepdim=True).values
    ymin = torch.min(corners[:, 1], dim=0, keepdim=True).values
    ymax = torch.max(corners[:, 1], dim=0, keepdim=True).values

    return torch.stack((xmin, ymin, xmax, ymax), dim=1)


def get_alpha(rot):
    """
    output: (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos,
                    bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    return rot[:, 0]
    """
    idx = (rot[:, 1] > rot[:, 5]).float()
    alpha1 = torch.arctan2(rot[:, 2], rot[:, 3]) + (-0.5 * torch.pi)
    alpha2 = torch.arctan2(rot[:, 6], rot[:, 7]) + (0.5 * torch.pi)
    return alpha1 * idx + alpha2 * (1 - idx)


def decoder(center_e, offset_e, size_e=None, K=60,
            attr_fn=None,
            size_scale=100.0,          # vestigial: attr_fn scales itself
            nms_kernel=3, nms_radius=0.0, nms_iou=0.0,
            nms_cell_size_cm=None, nms_score_threshold=0.0):
    """
    center_e: B,1,H,W (post-sigmoid scores)
    offset_e: B,4,H,W (channels 0-1: sub-pixel, channels 2-3: temporal)

    Per-object 3D attributes (NEW):
        attr_fn : callable or None. Receives the decoded SUB-PIXEL peak
                  positions xy (B,K,2) in BEV memory coords and returns a
                  dict with keys
                      'yaw_sincos'    (B,K,2)
                      'yaw_angle'     (B,K)
                      'dimensions'    (B,K,3) CENTIMETRES
                      'posture_prob'  (B,K)
                      'posture_class' (B,K)
                  -- exactly what Attr3DQueryHead.query_extra() produces.
                  Attributes are evaluated ONCE per decoded peak
                  (per-object), not per pixel.

    Duplicate suppression (unchanged semantics):
        nms_kernel / nms_radius / nms_iou / nms_cell_size_cm /
        nms_score_threshold as before. The IoU gate now consumes the
        per-peak attributes from attr_fn instead of dense maps.

    Suppressed peaks are NOT removed (tensor shapes stay B,K) -- their
    score is set to 0.0, so every existing `score > conf_threshold`
    filter downstream drops them automatically.

    Returns
    -------
    (xy, xy_prev, scores, clses)          if attr_fn is None
    (xy, xy_prev, scores, clses, extra)   otherwise
    """
    batch, cat, height, width = center_e.size()

    center_e = _nms(center_e, kernel=nms_kernel)

    topk_scores, topk_inds = torch.topk(
        center_e.view(batch, cat, -1), K
    )
    topk_inds = topk_inds % (height * width)

    ys = (topk_inds // width).float()
    xs = (topk_inds % width).float()

    scores, topk_ind = torch.topk(
        topk_scores.view(batch, -1), K
    )
    clses = (topk_ind // K).int()

    # ── re-gather topk_inds to get true spatial indices ──
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind
    ).view(batch, K)

    offset = _transpose_and_gather_feat(offset_e, topk_inds)
    ys = _gather_feat(ys.view(batch, -1, 1), topk_ind).view(batch, K)
    xs = _gather_feat(xs.view(batch, -1, 1), topk_ind).view(batch, K)

    xs_int = xs.view(batch, K, 1)
    ys_int = ys.view(batch, K, 1)

    xs = xs_int + offset[:, :, 0:1]
    ys = ys_int + offset[:, :, 1:2]
    xy = torch.cat((xs, ys), dim=2)

    xs_prev = xs_int + offset[:, :, 2:3]
    ys_prev = ys_int + offset[:, :, 3:4]
    xy_prev = torch.cat((xs_prev, ys_prev), dim=2)

    # per-object 3D attributes at the decoded peaks
    # xy already includes the predicted sub-pixel offset, i.e. it is the
    # SAME (sub-pixel) construct the query head is trained on at GT
    # centres -- no integer-cell rounding anywhere.
    extra = {}
    if attr_fn is not None:
        with torch.no_grad():
            extra = {k: v.detach() for k, v in attr_fn(xy).items()}

    # second-stage NMS over the surviving Top-K peaks
    if (nms_radius and nms_radius > 0) or (nms_iou and nms_iou > 0):
        boxes = None
        if (nms_iou and nms_iou > 0 and nms_cell_size_cm
                and 'dimensions' in extra and 'yaw_angle' in extra):
            cell = float(nms_cell_size_cm)
            boxes = torch.stack((
                xy[..., 0], xy[..., 1],
                extra['dimensions'][..., 0] / cell,   # length -> cells
                extra['dimensions'][..., 1] / cell,   # width  -> cells
                extra['yaw_angle'],
            ), dim=-1)                                             # B,K,5
        scores = scores.detach().clone()
        for b in range(batch):
            dup = _bev_nms_mask(
                xy[b], scores[b],
                radius=float(nms_radius or 0.0),
                boxes=None if boxes is None else boxes[b],
                iou_thresh=float(nms_iou or 0.0),
                score_thresh=float(nms_score_threshold or 0.0),
            )
            scores[b] = scores[b].masked_fill(dup, 0.0)

    if extra:
        return (xy.detach(), xy_prev.detach(), scores.detach(),
                clses.detach(), extra)
    return xy.detach(), xy_prev.detach(), scores.detach(), clses.detach()


def _topk(scores, K=40):
    batch, cat, length, width = scores.size()  # cat = 1

    '''
    For each channel, select K positions with high scores, cat * K total positions
    topk_scores / topk_inds: (bs, cat, K)

    From cat * K positions, select K positions with high scores
    topk_score / topk_ind: (bs, K)
    topk_clses: (bs, K)
    '''
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    # topk_inds = topk_inds % (length * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_ys, topk_xs


def _gather_feat(feat, ind, mask=None):
    # feat: (bs, h*w, 2), ind: (bs, max_objs)
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)  # (bs, max_objs, 2)
    feat = feat.gather(1, ind)  # (bs, max_objs, 2)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()  # (bs, 2, 56, 56) -> (bs, 56, 56, 2)
    feat = feat.view(feat.size(0), -1, feat.size(3))  # (bs, 56*56, 2)
    feat = _gather_feat(feat, ind)  # (bs, max_objs, 2)
    return feat
