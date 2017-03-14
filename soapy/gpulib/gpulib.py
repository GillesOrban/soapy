from numba import cuda
import numba
import numpy
import math
# Cuda threads per block
CUDA_TPB = 32

def linterp2d(data, xCoords, yCoords, interpArray, threadsPerBlock=None):
    """
    2-D interpolation using purely python - fast if compiled with numba
    Parameters:
        array (ndarray): The 2-D array to interpolate
        xCoords (ndarray): A 1-D array of x-coordinates
        yCoords (ndarray): A 2-D array of y-coordinates
        interpArray (ndarray): The array to place the calculation
    Returns:
        interpArray (ndarray): A pointer to the calculated ``interpArray''
    """
    if threadsPerBlock is None:
        threadsPerBlock = CUDA_TPB

    tpb = (threadsPerBlock, )*2
    # blocks per grid
    bpg = (
            numpy.ceil(interpArray.shape[0]/tpb),
            numpy.ceil(interpArray.shape[1]/tpb)
            )

    linterp2d_kernel[tpb, bpg](data, xCoords, yCoords, interpArray)

    return interpArray

@cuda.jit
def linterp2d_kernel(data, xCoords, yCoords, interpArray):
    """
    2-D interpolation using purely python - fast if compiled with numba
    Parameters:
        array (ndarray): The 2-D array to interpolate
        xCoords (ndarray): A 1-D array of x-coordinates
        yCoords (ndarray): A 2-D array of y-coordinates
        interpArray (ndarray): The array to place the calculation
    Returns:
        interpArray (ndarray): A pointer to the calculated ``interpArray''
    """
    # Thread id in a 1D block
    i, j = cuda.grid(2)

    # Get corresponding coordinates in image
    x = xCoords[i]
    x1 = numba.int32(x)
    y = yCoords[j]
    y1 = numba.int32(y)

    # Do bilinear interpolation
    xGrad1 = data[x1+1, y1] - data[x1, y1]
    a1 = data[x1, y1] + xGrad1*(x-x1)

    xGrad2 = data[x1+1, y1+1] - data[x1, y1+1]
    a2 = data[x1, y1+1] + xGrad2*(x-x1)

    yGrad = a2 - a1
    interpArray[i,j] = a1 + yGrad*(y-y1)

def bilinterp2d_regular(
        data, xMin, xMax, xSize, yMin, yMax, ySize, interpArray, threadsPerBlock=None):
    """
    2-D interpolation on a regular grid using purely python - 
    fast if compiled with numba
    Parameters:
        array (ndarray): The 2-D array to interpolate
        xCoords (ndarray): A 1-D array of x-coordinates
        yCoords (ndarray): A 2-D array of y-coordinates
        interpArray (ndarray): The array to place the calculation
    Returns:
        interpArray (ndarray): A pointer to the calculated ``interpArray''
    """
    if threadsPerBlock is None:
        threadsPerBlock = CUDA_TPB

    tpb = (threadsPerBlock, )*2
    # blocks per grid
    bpg = (
            int(numpy.ceil(interpArray.shape[0]/threadsPerBlock)),
            int(numpy.ceil(interpArray.shape[1]/threadsPerBlock))
            )
    bilinterp2d_regular_kernel[tpb, bpg](data, xMin, xMax, xSize, yMin, yMax, ySize, interpArray)

    return interpArray

@cuda.jit
def bilinterp2d_regular_kernel(
        data, xMin, xMax, xSize, yMin, yMax, ySize, interpArray):
    """
    2-D interpolation using purely python - fast if compiled with numba
    Parameters:
        array (ndarray): The 2-D array to interpolate
        xCoords (ndarray): A 1-D array of x-coordinates
        yCoords (ndarray): A 2-D array of y-coordinates
        interpArray (ndarray): The array to place the calculation
    Returns:
        interpArray (ndarray): A pointer to the calculated ``interpArray''
    """
    # Thread id in a 2D grid
    i, j = cuda.grid(2)

    x = xMin + i*float(xMax - xMin)/(xSize - 1)
    x1 = numba.int32(x)

    y = yMin + j*float(yMax - yMin)/(ySize - 1)
    y1 = numba.int32(y)

    xGrad1 = data[x1+1, y1] - data[x1, y1]
    a1 = data[x1, y1] + xGrad1*(x-x1)

    xGrad2 = data[x1+1, y1+1] - data[x1, y1+1]
    a2 = data[x1, y1+1] + xGrad2*(x-x1)

    yGrad = a2 - a1
    interpArray[i,j] += a1 + yGrad*(y-y1)


def zoom(data, zoomArray, threadsPerBlock=None):
    """
    2-D zoom interpolation using purely python - fast if compiled with numba.
    Both the array to zoom and the output array are required as arguments, the
    zoom level is calculated from the size of the new array.
    Parameters:
        array (ndarray): The 2-D array to zoom
        zoomArray (ndarray): The array to place the calculation
    Returns:
        ndarray: A pointer to the zoomArray
    """
    if threadsPerBlock is None:
        threadsPerBlock = CUDA_TPB

    tpb = (threadsPerBlock, )*2
    # blocks per grid
    bpg = (
            numpy.ceil(float(zoomArray.shape[0])/tpb),
            numpy.ceil(float(zoomArray.shape[1])/tpb)
            )

    zoom_kernel[tpb, bpg](data, zoomArray)

    return zoomArray

@cuda.jit
def zoom_kernel(data, zoomArray):
    """
    2-D zoom interpolation using purely python - fast if compiled with numba.
    Both the array to zoom and the output array are required as arguments, the
    zoom level is calculated from the size of the new array.
    Parameters:
        array (ndarray): The 2-D array to zoom
        zoomArray (ndarray): The array to place the calculation
    """
    i, j = cuda.grid(2)

    x = i*numba.float32(data.shape[0]-1)/(zoomArray.shape[0]-0.99999999)
    x1 = numba.int32(x)

    y = j*numba.float32(data.shape[1]-1)/(zoomArray.shape[1]-0.99999999)
    y1 = numba.int32(y)

    xGrad1 = data[x1+1, y1] - data[x1, y1]
    a1 = data[x1, y1] + xGrad1*(x-x1)

    xGrad2 = data[x1+1, y1+1] - data[x1, y1+1]
    a2 = data[x1, y1+1] + xGrad2*(x-x1)

    yGrad = a2 - a1
    zoomArray[i,j] = a1 + yGrad*(y-y1)


def phs2EField(phase, EField):
    """
    Converts phase to an efield on the GPU
    """
    if threadsPerBlock is None:
        threadsPerBlock = CUDA_TPB

    tpb = (threadsPerBlock, )*2
    # blocks per grid
    bpg = (
            numpy.ceil(float(phase.shape[0])/tpb),
            numpy.ceil(float(phase.shape[1])/tpb)
            )

    phs2EField_kernel[tpb, bpg](phase, EField)

    return EField

@cuda.jit
def phs2EField_kernel(phase, EField):
    i, j = cuda.grid(2)

    EField[i, j] = math.exp(phs[i, j])

def absSquared3d(inputData, outputData, threadsPerBlock=None):

    if threadsPerBlock is None:
        threadsPerBlock = CUDA_TPB

    tpb = (threadsPerBlock,)*3
    # blocks per grid
    bpg = (
            int(numpy.ceil(float(inputData.shape[0])/threadsPerBlock)),
            int(numpy.ceil(float(inputData.shape[1])/threadsPerBlock)),
            int(numpy.ceil(float(inputData.shape[2])/threadsPerBlock))
            )

    absSquared3d_kernel[tpb, bpg](inputData, outputData)

    return outputData

@cuda.jit
def absSquared3d_kernel(inputData, outputData):
    i, j, k = cuda.grid(3)
    outputData[i, j, k] = inputData[i, j, k].real**2 + inputData[i, j, k].imag**2


def array_sum(array1, array2, threadsPerBlock=None):

    if threadsPerBlock is None:
        threadsPerBlock = CUDA_TPB

    tpb = threadsPerBlock
    # blocks per grid
    bpg = int(numpy.ceil(float(array1.shape[0])/threadsPerBlock)),


    array_sum_kernel[tpb, bpg](array1, array2)

    return array1

@cuda.jit
def array_sum_kernel(array1, array2):
    i = cuda.grid(1)

    array1[i] += array2[i]


def array_sum2d(array1, array2, threadsPerBlock=None):

    if threadsPerBlock is None:
        threadsPerBlock = CUDA_TPB

    tpb = (threadsPerBlock, )*2
    # blocks per grid
    bpg = (
            int(numpy.ceil(float(array1.shape[0])/threadsPerBlock)),
            int(numpy.ceil(float(array1.shape[1])/threadsPerBlock))
            )

    array_sum2d_kernel[tpb, bpg](array1, array2)

    return array1

@cuda.jit
def array_sum2d_kernel(array1, array2):
    i, j = cuda.grid(2)

    if i < array1.shape[0]:
        if j < array1.shape[1]:
            array1[i, j] += array2[i, j]
