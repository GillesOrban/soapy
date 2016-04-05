from soapy import confParse, aoSimLib, DM, WFS, gpulib
import unittest
import numpy
from numba import cuda
import os
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../conf/")

class testGpuWfs(unittest.TestCase):

    def testMaskCrop2Subaps(self):
        PAD = 10
        # Make simple pupil
        mask = aoSimLib.circle(64, 128).astype('int32')
        # pad like in sim
        mask = numpy.pad(mask, ((PAD, PAD),(PAD, PAD)), mode='constant')
        EField = mask.astype('complex64')

        # find coords for 4x4 subaps
        subapCoords = numpy.zeros((16, 2))
        for x in range(4):
            for y in range(4):
                subapCoords[x*4 + y] = x*32, y*32

        # CPU version #######################
        # Apply the scaled pupil mask
        EField *= mask

        # Now cut out only the eField across the pupilSize
        cropEField = EField[PAD:-PAD, PAD:-PAD]

        # Create an array of individual subap EFields
        subapArrays_cpu = numpy.zeros((16, 32, 32)).astype('complex64')
        for i, coord in enumerate(subapCoords):
            x, y = coord
            subapArrays_cpu[i] = cropEField[
                                    int(x):
                                    int(x+32) ,
                                    int(y):
                                    int(y+32)]
        #################################

        # Now do the same for the GPU...
        subapArrays_gpu = cuda.device_array_like(
                subapArrays_cpu.astype('complex64'))
        coords_gpu = cuda.to_device(subapCoords.astype('float32'))
        EField_gpu = cuda.to_device(EField.astype('complex64'))
        mask_gpu = cuda.to_device(mask.astype('int32'))
        tiltFix = cuda.to_device(numpy.ones((16, 16)).astype('complex64'))

        gpulib.wfs.maskCrop2Subaps(
                subapArrays_gpu, EField_gpu, mask_gpu, 16, coords_gpu, tiltFix)

        print(subapArrays_cpu, subapArrays_gpu)

        assert numpy.allclose(subapArrays_cpu, subapArrays_gpu.copy_to_host())


    def testAbsSquared3d(self):
        shape = (10, 20, 20)

        a = (numpy.random.random(shape)
                + 1j *  numpy.random.random(shape)).astype('complex64')

        a_gpu = cuda.to_device(a)
        b_gpu = cuda.device_array(shape, dtype='float32')

        # CPU
        b = abs(a)**2

        # GPUs
        gpulib.absSquared3d(a_gpu, b_gpu)

        assert(numpy.allclose(b_gpu.copy_to_host(), b))
