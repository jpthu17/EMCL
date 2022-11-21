class AverageMeter(object):
  def __init__(self):
    self.dic = {}
    self.reset()

  def reset(self):
    for key in self.dic:
      for metric in self.dic[key]:
        self.dic[key][metric] = 0

  def update(self, key, val, n=1):
    self.dic.setdefault(key, {'val': 0, 'sum': 0, 'count': 0, 'avg': 0})
    self.dic[key]['val'] = val
    self.dic[key]['sum'] += val * n
    self.dic[key]['count'] += n
    self.dic[key]['avg'] = self.dic[key]['sum'] / self.dic[key]['count']
