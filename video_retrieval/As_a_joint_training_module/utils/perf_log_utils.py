import time


def update_perf_log(epoch_perf, perf_log_path):
  now = time.strftime('%c')
  line = 't: {}, '.format(now)
  for key in epoch_perf:
    line += '{}: {}, '.format(key, epoch_perf[key])

  line += '\n'

  with open(perf_log_path, 'a') as file:
    file.write(line)
