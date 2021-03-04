import tensorflow as tf
import math
import numpy as np
import unittest

lg_e_10 = math.log(10)
def log10(x):
    """Convert a new tensor with the base-10 logarithm of the elements of x. """
    return tf.math.log(x) / lg_e_10

class Result(object):
    def __init__(self):
        self.irmse = 0
        self.imae = 0
        self.mse = 0
        self.rmse = 0
        self.mae = 0
        self.absrel = 0
        self.squared_rel = 0
        self.lg10 = 0
        self.delta1 = 0
        self.delta2 = 0
        self.delta3 = 0
        self.data_time = 0
        self.gpu_time = 0
        self.silog = 0 # Scale invariant logarithmic error [log(m)*100]
        self.photometric = 0
        self.count=0.0

    def set_to_worst(self):
        self.irmse = np.inf
        self.imae = np.inf
        self.mse = np.inf
        self.rmse = np.inf
        self.mae = np.inf
        self.absrel = np.inf
        self.squared_rel = np.inf
        self.lg10 = np.inf
        self.silog = np.inf
        self.delta1 = 0
        self.delta2 = 0
        self.delta3 = 0
        self.data_time = 0
        self.gpu_time = 0

        
    def finalize(self):
        self.irmse = self.irmse/self.count
        self.imae = self.imae/self.count
        self.mse = self.mse/self.count
        self.rmse = self.rmse/self.count
        self.mae = self.mae/self.count
        self.absrel = self.absrel/self.count
        self.squared_rel = self.squared_rel/self.count
        self.lg10 = self.lg10/self.count
        self.delta1 = self.delta1/self.count
        self.delta2 = self.delta2/self.count
        self.delta3 = self.delta3/self.count
        self.data_time = self.data_time/self.count
        self.gpu_time = self.gpu_time/self.count
        self.silog = self.silog/self.count
        self.photometric = self.photometric/self.count

    def update(self, irmse, imae, mse, rmse, mae, absrel, squared_rel, lg10, delta1, delta2, delta3, gpu_time, data_time, silog, photometric=0):
        self.count+=1.0
        self.irmse += irmse
        self.imae += imae
        self.mse += mse
        self.rmse += rmse
        self.mae += mae
        self.absrel += absrel
        self.squared_rel += squared_rel
        self.lg10 += lg10
        self.delta1 += delta1
        self.delta2 +=delta2
        self.delta3 += delta3
        self.data_time += data_time
        self.gpu_time += gpu_time
        self.silog += silog
        self.photometric += photometric

    def evaluate(self, output, target, photometric=0):
        #valid_mask = target>0.1
        #valid_mask = target>0.1
        valid_mask_1 = output>1.0 
        valid_mask_2 = target>0.1
        valid_mask=np.logical_and(valid_mask_1,valid_mask_2)
        # convert from meters to mm
        output_mm = 1e3 * output[valid_mask]
        target_mm = 1e3 * target[valid_mask]

        abs_diff = np.abs(output_mm - target_mm)

        self.mse = np.mean((np.power(abs_diff, 2)))
        self.rmse = math.sqrt(self.mse)
        self.mae = np.mean(abs_diff)
        '''
        self.lg10 = np.mean(np.abs(log10(output_mm) - log10(target_mm)))
        self.absrel = np.mean((abs_diff / target_mm))
        self.squared_rel =np.mean(((abs_diff / target_mm) ** 2))

        maxRatio = tf.math.maximum(output_mm / target_mm, target_mm / output_mm)
        self.delta1 = np.mean((maxRatio < 1.25))
        self.delta2 = np.mean((maxRatio < 1.25 ** 2))
        self.delta3 = np.mean((maxRatio < 1.25 ** 3))
        self.data_time = 0
        self.gpu_time = 0

        # silog uses meters
        err_log = tf.math.log(target[valid_mask]) - tf.math.log(output[valid_mask])
        normalized_squared_log = np.mean(err_log ** 2)
        log_mean = np.mean(err_log)
        self.silog = math.sqrt(normalized_squared_log - log_mean*log_mean)*100
        
       '''
        # convert from meters to km
        inv_output_km = (1e-3 * output[valid_mask]) ** (-1)
        inv_target_km = (1e-3 * target[valid_mask]) ** (-1)
        abs_inv_diff = tf.abs(inv_output_km - inv_target_km)
        self.irmse = math.sqrt(tf.reduce_mean(tf.math.pow(abs_inv_diff, 2)))
        self.imae = np.mean(abs_inv_diff)

        self.photometric = float(photometric)
 
