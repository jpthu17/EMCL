import ipdb
import numpy as np
import scipy.stats


def t2v_metrics(sims, query_masks=None):
  assert sims.ndim == 2, "expected a matrix"
  nq, nv = sims.shape
  dists = -sims
  sorted_dists = np.sort(dists, axis=1)

  qu = nq // nv  # Nb of queries per video
  gt_idx = [[
      np.ravel_multi_index([ii, jj], (nq, nv))
      for ii in range(jj * qu, (jj + 1) * qu)
  ]
            for jj in range(nv)]
  gt_idx = np.array(gt_idx)
  gt_dists = dists.reshape(-1)[gt_idx.reshape(-1)]
  gt_dists = gt_dists[:, np.newaxis]
  rows, cols = np.where(
      (sorted_dists - gt_dists) == 0)  # find column position of GT
  break_ties = "averaging"

  if rows.size > nq:
    assert np.unique(rows).size == nq, "issue in metric evaluation"
    if break_ties == "optimistically":
      _, idx = np.unique(rows, return_index=True)
      cols = cols[idx]
    elif break_ties == "averaging":
      # fast implementation, based on this code:
      # https://stackoverflow.com/a/49239335
      locs = np.argwhere((sorted_dists - gt_dists) == 0)

      # Find the split indices
      steps = np.diff(locs[:, 0])
      splits = np.nonzero(steps)[0] + 1
      splits = np.insert(splits, 0, 0)

      # Compute the result columns
      summed_cols = np.add.reduceat(locs[:, 1], splits)
      counts = np.diff(np.append(splits, locs.shape[0]))
      avg_cols = summed_cols / counts
      cols = avg_cols

  msg = "expected ranks to match queries ({} vs {}) "
  if cols.size != nq:
    ipdb.set_trace()
  assert cols.size == nq, msg

  if query_masks is not None:
    # remove invalid queries
    assert query_masks.size == nq, "invalid query mask shape"
    cols = cols[query_masks.reshape(-1).astype(np.bool)]
    assert cols.size == query_masks.sum(), "masking was not applied correctly"
    # update number of queries to account for those that were missing
    nq = query_masks.sum()

  return cols2metrics(cols, nq)


def v2t_metrics(sims, query_masks=None):
  sims = sims.T

  assert sims.ndim == 2, "expected a matrix"
  num_queries, num_caps = sims.shape
  dists = -sims
  caps_per_video = num_caps // num_queries
  break_ties = "averaging"

  missing_val = 1E8
  query_ranks = []
  for ii in range(num_queries):
    row_dists = dists[ii, :]
    if query_masks is not None:
      row_dists[np.logical_not(query_masks.reshape(-1))] = missing_val
    sorted_dists = np.sort(row_dists)

    min_rank = np.inf
    for jj in range(ii * caps_per_video, (ii + 1) * caps_per_video):
      if row_dists[jj] == missing_val:
        # skip rankings of missing captions
        continue
      ranks = np.where((sorted_dists - row_dists[jj]) == 0)[0]
      if break_ties == "optimistically":
        rank = ranks[0]
      elif break_ties == "averaging":
        rank = ranks.mean()
      if rank < min_rank:
        min_rank = rank
    query_ranks.append(min_rank)
  query_ranks = np.array(query_ranks)

  return cols2metrics(query_ranks, num_queries)


def cols2metrics(cols, num_queries):
  """Compute the metrics."""
  metrics = {}
  metrics["R1"] = 100 * float(np.sum(cols == 0)) / num_queries
  metrics["R5"] = 100 * float(np.sum(cols < 5)) / num_queries
  metrics["R10"] = 100 * float(np.sum(cols < 10)) / num_queries
  metrics["R50"] = 100 * float(np.sum(cols < 50)) / num_queries
  metrics["MedR"] = np.median(cols) + 1
  metrics["MeanR"] = np.mean(cols) + 1
  stats = [metrics[x] for x in ("R1", "R5", "R10")]
  metrics["geometric_mean_R1-R5-R10"] = scipy.stats.mstats.gmean(stats)
  metrics["cols"] = [int(i) for i in list(cols)]
  return metrics
