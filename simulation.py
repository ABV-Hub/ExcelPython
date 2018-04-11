from __future__ import division
import sys
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xlwings as xw

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

sht = xw.Book.caller().sheets[0]
gridVaryOpt={ }
defaultDim = 200

    
def readGeneralInfo():
    nx = sht.range('B3').options(numbers=int).value
    ny = sht.range('C3').options(numbers=int).value
    nz = sht.range('D3').options(numbers=int).value

    gridVaryOpt['xopt'] = sht.range('B4').value
    gridVaryOpt['yopt'] = sht.range('C4').value
    gridVaryOpt['zopt'] = sht.range('D4').value

    gridVaryOpt['por'] = sht.range('F3').value
    gridVaryOpt['kx'] = sht.range('G3').value
    gridVaryOpt['ky'] = sht.range('H3').value
    gridVaryOpt['kz'] = sht.range('I3').value

    return nx,ny,nz

nx,ny,nz = readGeneralInfo()

def createHDF5(nx,ny,nz):
    
    ni = nx
    nj = ny
    nk = nz

    fModel = h5py.File(".\\simModel.hdf5", "w")
    fModel.create_group("StaticGroup")
    fModel.create_group("DynamicGroup")
    fModel.create_group("PvtGroup")
    fModel.create_dataset('StaticGroup/dx', (ni,nj,nk), dtype='f')
    fModel.create_dataset('StaticGroup/dy', (ni,nj,nk), dtype='f')
    fModel.create_dataset('StaticGroup/dz', (ni,nj,nk), dtype='f')
    fModel.create_dataset('StaticGroup/por', (ni,nj,nk), dtype='f')
    fModel.create_dataset('StaticGroup/permx', (ni,nj,nk), dtype='f')
    fModel.create_dataset('StaticGroup/permy', (ni,nj,nk), dtype='f')
    fModel.create_dataset('StaticGroup/permz', (ni,nj,nk), dtype='f')

    fModel.close()

def init():

    sht.range((9,3),(9,defaultDim+3-1)).color = (255,255,255)
    sht.range((10,3),(10,defaultDim+3-1)).color = (255,255,255)
    sht.range((11,3),(11,defaultDim+3-1)).color = (255,255,255)
    sht.range((17,3),(17,defaultDim+3-1)).color = (255,255,255)
    sht.range((18,3),(18,defaultDim+3-1)).color = (255,255,255)
    sht.range((19,3),(19,defaultDim+3-1)).color = (255,255,255)
    sht.range((20,3),(20,defaultDim+3-1)).color = (255,255,255)


def defineGrid():
    init()

    fig = plt.figure()
    ax = fig.gca(projection='3d')


# Make data.
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = np.sin(R)

# Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

    createHDF5(nx,ny,nz)

#prepare gui    
    if gridVaryOpt['xopt'] == 'CON':
        sht.range('A9').value = 'DX'
        sht.range('B9').value = 'CON'
        sht.range('C9').color = (102,255,102)
    elif gridVaryOpt['xopt'] == 'XVAR':
        sht.range('A9').value = 'DX'
        sht.range('B9').value = 'XVAR'
        sht.range((9,3),(9,nx+3-1)).color = (102,255,102)

    if gridVaryOpt['yopt'] == 'CON':
        sht.range('A10').value = 'DY'
        sht.range('B10').value = 'CON'
        sht.range('C10').color = (102,255,102)
    elif gridVaryOpt['yopt'] == 'YVAR':
        sht.range('A10').value = 'DY'
        sht.range('B10').value = 'YVAR'
        sht.range((10,3),(10,ny+3-1)).color = (102,255,102)
        
    if gridVaryOpt['zopt'] == 'CON':
        sht.range('A11').value = 'DZ'
        sht.range('B11').value = 'CON'
        sht.range('C11').color = (102,255,102)
    elif gridVaryOpt['zopt'] == 'ZVAR':
        sht.range('A11').value = 'DZ'
        sht.range('B11').value = 'ZVAR'
        sht.range((11,3),(11,nz+3-1)).color = (102,255,102)

    if gridVaryOpt['por'] == 'CON':
        sht.range('A17').value = 'POR'
        sht.range('B17').value = 'CON'
        sht.range('C17').color = (102,255,102)
    elif gridVaryOpt['por'] == 'ZVAR':
        sht.range('A17').value = 'POR'
        sht.range('B17').value = 'ZVAR'
        sht.range((17,3),(17,nz+3-1)).color = (102,255,102)

    if gridVaryOpt['kx'] == 'CON':
        sht.range('A18').value = 'PERMX'
        sht.range('B18').value = 'CON'
        sht.range('C18').color = (102,255,102)
    elif gridVaryOpt['kx'] == 'ZVAR':
        sht.range('A18').value = 'PERMX'
        sht.range('B18').value = 'ZVAR'
        sht.range((18,3),(18,nz+3-1)).color = (102,255,102)
    
    if gridVaryOpt['ky'] == 'CON':
        sht.range('A19').value = 'PERMY'
        sht.range('B19').value = 'CON'
        sht.range('C19').color = (102,255,102)
    elif gridVaryOpt['ky'] == 'ZVAR':
        sht.range('A19').value = 'PERMY'
        sht.range('B19').value = 'ZVAR'
        sht.range((19,3),(19,nz+3-1)).color = (102,255,102)

    if gridVaryOpt['kz'] == 'CON':
        sht.range('A20').value = 'PERMZ'
        sht.range('B20').value = 'CON'
        sht.range('C20').color = (102,255,102)
    elif gridVaryOpt['kz'] == 'ZVAR':
        sht.range('A20').value = 'PERMZ'
        sht.range('B20').value = 'ZVAR'
        sht.range((20,3),(20,nz+3-1)).color = (102,255,102)

def readGrid():

    fModel = h5py.File(".\\simModel.hdf5", "a")
    dset_dx = fModel['StaticGroup/dx']
    dset_dy = fModel['StaticGroup/dy']
    dset_dz = fModel['StaticGroup/dz']
    dset_por = fModel['StaticGroup/por']
    dset_kx = fModel['StaticGroup/permx']
    dset_ky = fModel['StaticGroup/permy']
    dset_kz = fModel['StaticGroup/permz']

    if gridVaryOpt['xopt'] == 'XVAR' :
        dxVar = sht.range((9,3),(9,nx+3-1)).options(np.array).value
        idArray = np.arange(nx)
        for id in idArray:
            dset_dx[id,:,:] = dxVar[id]
    elif gridVaryOpt['xopt'] == 'CON' :
        dset_dx[:,:,:] = sht.range((9,3)).options(numbers=float).value

    if gridVaryOpt['yopt'] == 'YVAR' :
        dyVar = sht.range((10,3),(10,ny+3-1)).options(np.array).value
        idArray = np.arange(ny)
        for id in idArray:
            dset_dy[:,id,:] = dyVar[id]
    elif gridVaryOpt['yopt'] == 'CON' :
        dset_dy[:,:,:] = sht.range((10,3)).options(numbers=float).value

    if gridVaryOpt['zopt'] == 'ZVAR' :
        dzVar = sht.range((11,3),(11,nz+3-1)).options(np.array).value
        idArray = np.arange(nz)
        for id in idArray:
            dset_dz[:,:,id] = dzVar[id]
    elif gridVaryOpt['zopt'] == 'CON' :
        dset_dz[:,:,:] = sht.range((11,3)).options(numbers=float).value

    if gridVaryOpt['por'] == 'ZVAR' :
        porVar = sht.range((17,3),(17,nz+3-1)).options(np.array).value
        idArray = np.arange(nz)
        for id in idArray:
            dset_por[:,:,id] = porVar[id]
    elif gridVaryOpt['por'] == 'CON' :
        dset_por[:,:,:] = sht.range((17,3)).options(numbers=float).value

    if gridVaryOpt['kx'] == 'ZVAR' :
        kxVar = sht.range((18,3),(18,nz+3-1)).options(np.array).value
        idArray = np.arange(nz)
        for id in idArray:
            dset_kx[:,:,id] = kxVar[id]
    elif gridVaryOpt['kx'] == 'CON' :
        dset_kx[:,:,:] = sht.range((18,3)).options(numbers=float).value

    if gridVaryOpt['ky'] == 'ZVAR' :
        kyVar = sht.range((19,3),(19,nz+3-1)).options(np.array).value
        idArray = np.arange(nz)
        for id in idArray:
            dset_ky[:,:,id] = kyVar[id]
    elif gridVaryOpt['ky'] == 'CON' :
        dset_ky[:,:,:] = sht.range((19,3)).options(numbers=float).value

    if gridVaryOpt['kz'] == 'ZVAR' :
        kzVar = sht.range((20,3),(20,nz+3-1)).options(np.array).value
        idArray = np.arange(nz)
        for id in idArray:
            dset_kz[:,:,id] = kzVar[id]
    elif gridVaryOpt['kz'] == 'CON' :
        dset_kz[:,:,:] = sht.range((20,3)).options(numbers=float).value

    fModel.flush()
    fModel.close()

def readPvt():
    df_PVTo = sht.range('B28:E39').options(pd.DataFrame).value
    # df to hdf5
    df_to_nparray = df_PVTo.to_records(index=True)
    f = h5py.File('..\\simModel.hdf5','w')

    # create dataset
    f['PvtGroup/PVTO'] = df_to_nparray

    # close connection to file
    f.close()
    
    #sht.range('G29:J39').options(pd.DataFrame).value = df2

def main():
    #sht = xw.Book.caller().sheets[0]
    # User Inputs
    # Grid section
    nx = sht.range('B3').options(numbers=int).value
    ny = sht.range('C3').options(numbers=int).value
    nz = sht.range('D3').options(numbers=int).value

    dx_condition = sht.range('B8').value
    if dx_condition == 'CON':
        dx = sht.range('C8').options(numbers=float).value

    dy_condition = sht.range('B9').value
    if dy_condition == 'CON':
        dy = sht.range('C9').options(numbers=float).value

    dz_condition = sht.range('B10').value
    if dz_condition == 'CON':
        dz = sht.range('C10').options(numbers=float).value
    
    tops_condition = sht.range('B12').value
    if tops_condition == 'CON':
        tops = sht.range('C12').options(numbers=float).value

    poro_condition = sht.range('B16').value
    if poro_condition == 'CON':
        poro = sht.range('C16').options(numbers=float).value
        
    permx_condition = sht.range('B17').value
    if permx_condition == 'CON':
        permx = sht.range('C17').options(numbers=float).value

    permy_condition = sht.range('B18').value
    if permy_condition == 'CON':
        permy = sht.range('C18').options(numbers=float).value

    permz_condition = sht.range('B19').value
    if permz_condition == 'CON':
        permz = sht.range('C19').options(numbers=float).value
    
    w_o_rpKW = sht.range('A21').value
    
    if w_o_rpKW == 'WATER-OIL':
        cnt = 1
        while True:
            swid = 'B'
            swid +=str(21+cnt)
            sw   = sht.range(swid).value
            
            if sw ==None:
                break
            else:
                cnt += 1
        wostart = 'B21'
        woend   = 'E'+str(21+cnt-1)
        woTableRange = wostart+':'+woend
        
        w_o_df = sht.range(woTableRange).options(pd.DataFrame,index=0,header=1).value
#        sw = w_o_df.as_matrix(['SAT'])
#        krw = w_o_df.as_matrix(['KRW'])
#        krow = w_o_df.as_matrix(['KROW'])

        fig = plt.figure()
        plt.plot([1,2,3])
        sht.pictures.add(fig)
