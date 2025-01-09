import torch
import torch.multiprocessing as mp
import time
import math


T_PI = torch.tensor(math.pi)

def mapBoundaries(pBound,
				  nElem,
				  valueLeftBound,
				  sizeElem,
				  offset):

	for k in range(nElem):
		pBound[k] = (k - valueLeftBound) * sizeElem + offset


def mapDet2Slice(pXmapp,
				 pYmapp,
				 tubeX,
				 tubeY,
				 tubeZ,
				 pXcoord,
				 pYcoord,
				 pZcoord,
				 ZSlicecoord,
				 nXelem,
				 nYelem,
				 numThreads):
	def process_row(x):
		for y in range(nYelem):
			ind = (x * nYelem) + y
			pXmapp[ind] = ((pXcoord[x] - tubeX) * (ZSlicecoord - pZcoord[y]) - (pXcoord[x] * tubeZ) + (
						pXcoord[x] * pZcoord[y])) / (-tubeZ + pZcoord[y])

			if x == 0:
				pYmapp[y] = ((pYcoord[y] - tubeY) * (ZSlicecoord - pZcoord[y]) - (pYcoord[y] * tubeZ) + (
							pYcoord[y] * pZcoord[y])) / (-tubeZ + pZcoord[y])

	with mp.Pool(numThreads) as mapPool:
		mapPool.map(process_row, range(nXelem))


def bilinear_interpolation(projI, pVolumet, pDetmX, pDetmY, nDetXMap, nDetYMap, nPixXMap, nPixYMap, dx, nz):
	pass


def projectionDDb(pProj,
	pVolume,
	pTubeAngle,
	pDetAngle,
	nProj,
	nPixX,
	nPixY,
	nSlices,
	nDetX,
	nDetY,
	idXProj,
	x_offset,
	y_offset,
	dx,
	dy,
	dz,
	du,
	dv,
	DSD,
	DDR,
	DAG):

	mp.set_start_method('spawn')
	nThreads = ...
	print(f"<device> running with max threads: {nThreads}")

	c_time = time.time()

	nDetXMap = nDetX + 1
	nDetYMap = nDetY + 1
	nPixXMap = nPixX + 1
	nPixYMap = nPixY + 1


	# Allocate memory contiguously
	projI = torch.zeros(nDetXMap*nDetYMap, dtype=torch.float64)


	# Pointer for projections coordinates
	# Allocate memory for projections coordinates
	pDetX = torch.zeros(nDetXMap, dtype=torch.float64)
	pDetY = torch.zeros(nDetYMap, dtype=torch.float64)
	pDetZ = torch.zeros(nDetYMap, dtype=torch.float64)
	pObjX = torch.zeros(nPixXMap, dtype=torch.float64)
	pObjY = torch.zeros(nPixYMap, dtype=torch.float64)
	pObjZ = torch.zeros(nSlices, dtype=torch.float64)


	# Pointer for mapped coordinates
	# Allocate memory for mapped coordinates
	pDetmY = torch.zeros(nDetYMap, dtype=torch.float64)
	pDetmX = torch.zeros(nDetYMap * nDetXMap, dtype=torch.float64)


	# Pointer for rotated detector coords
	# Allocate memory for rotated detector coords
	pRdetY = torch.zeros(nDetYMap, dtype=torch.float64)
	pRdetZ = torch.zeros(nDetYMap, dtype=torch.float64)


	# Map detector and object boundaries
	mapBoundaries(pDetX, nDetXMap, nDetX, -du, 0.0)
	
	mapBoundaries(pDetY, nDetYMap, nDetY / 2.0, dv, 0.0)
	
	mapBoundaries(pDetZ, nDetYMap, 0.0, 0.0, 0.0)
	
	mapBoundaries(pObjX, nPixXMap, nPixX, -dx, x_offset)
	
	mapBoundaries(pObjY, nPixYMap, nPixY / 2.0, dy, y_offset)
	
	mapBoundaries(pObjZ, nSlices, 0.0, dz, DAG + (dz / 2.0))


	# X - ray tube initial position
	tubeX = 0
	tubeY = 0
	tubeZ = DSD

	# Iso - center position
	isoY = 0
	isoZ = DDR


	# Allocate memory for temp projection variable
	pProjt = torch.zeros(nDetY * nDetX * nProj, dtype=torch.float64)
	pVolumet = torch.zeros(nPixYMap * nPixXMap * nSlices, dtype=torch.float64)


	# Integration of 2D slices over the whole volume
	# (S.1.Integration. - Liu et al(2017))
	"""
		2 - D image

		 -->J
		|	-------------
		v	|			|
		I	|			|
			|			|
			|			|
			-------------
	"""

	# Initialized everything to 0, so just hope that works
	# # Initialize first column and row with zeros
	# for nz in range(nSlices):
	# 	for y in range(nPixYMap):
	# 		pVolumet


	# Integrate on I direction
	def I_integrate_slice(x):
		i_sum = 0
		for y in range(nPixY):
			i_sum = i_sum + pVolume[(nz * nPixY * nPixX) + (x * nPixY) + y]
			pVolumet[(nz*nPixYMap *nPixXMap) + ((x+1)*nPixYMap) + y + 1] = i_sum

	for nz in range(nSlices):
		with mp.Pool(nThreads) as integrationPool:
			integrationPool.map(I_integrate_slice, range(nPixX))


	# Integrate on J direction
	def J_integrate_slice(y):
		for x in range(nPixXMap):
			pVolumet[(nz * nPixYMap * nPixXMap) + (x * nPixYMap) + y] += pVolumet[(nz * nPixYMap * nPixXMap) + ((x - 1) * nPixYMap) + y]

	for nz in range(nSlices):
		with mp.Pool(nThreads) as integrationPool:
			integrationPool.map(J_integrate_slice, range(1, nPixYMap))


	# Test if we will loop over all projs or not
	projIni = 0
	projEnd = nProj
	if idXProj != -1:
		projIni = idXProj
		projEnd = idXProj + 1


	# For each projection
	for p in range(projIni, projEnd):

		# Get specif tube angle for the projection
		theta = pTubeAngle[p] * T_PI / 180.0

		# Get specif detector angle for the projection
		phi = pDetAngle[p] * T_PI / 180.0

		# Tube rotation
		rtubeY = ((tubeY - isoY)*torch.cos(theta) - (tubeZ - isoZ)*torch.sin(theta)) + isoY
		rtubeZ = ((tubeY - isoY)*torch.sin(theta) + (tubeZ - isoZ)*torch.cos(theta)) + isoZ

		# Detector rotation
		for y in range(nDetYMap):
			pRdetY[y] = ((pDetY[y] - isoY)*torch.cos(phi) - (pDetZ[y] - isoZ)*torch.sin(phi)) + isoY
			pRdetZ[y] = ((pDetY[y] - isoY)*torch.sin(phi) + (pDetZ[y] - isoZ)*torch.cos(phi)) + isoZ


		# For each slice
			for nz in range(nSlices): {

				"""
				Map detector onto XY plane(Inside proj loop in case detector rotates)

				*** Note: Matlab has linear indexing as a column of elements, i.e, the elements are actually stored in memory as queued columns
				"""

				# Map slice onto XY plane
				mapDet2Slice(pDetmX, pDetmY, tubeX, rtubeY, rtubeZ, pDetX, pRdetY, pRdetZ, pObjZ[nz], nDetXMap, nDetYMap)

				"""
				S.2. Interpolation - Liu et al (2017)
				"""

				bilinear_interpolation(projI, pVolumet, pDetmX, pDetmY, nDetXMap, nDetYMap, nPixXMap, nPixYMap, dx, nz)

				"""
				S.3. Differentiation - Eq. 24 - Liu et al (2017)
				"""

				differentiation(pProjt, projI, pDetmX, pDetmY, tubeX, rtubeY, rtubeZ, pDetX, pRdetY, pRdetZ, nDetX, nDetY, nDetXMap, nDetYMap, du, dv, dx, dy, dz, p)




def projectionDDb_lib(pVolume,
					  pProj,
					  pGeo,
					  idXProj):
	nPixX = pGeo[0]
	nPixY = pGeo[1]
	nSlices = pGeo[2]
	nDetX = pGeo[3]
	nDetY = pGeo[4]

	dx = pGeo[5]
	dy = pGeo[6]
	dz = pGeo[7]
	du = pGeo[8]
	dv = pGeo[9]

	DSD = pGeo[10]
	DDR = pGeo[12]
	DAG = pGeo[14]

	nProj = pGeo[15]

	tubeAngle = pGeo[16]
	detAngle = pGeo[17]

	# # malloc statement in pyDBT allocates nProj contiguously in 64-bit chunks
	# pTubeAngle = torch.zeros(nProj, dtype=torch.float64)
	# pDetAngle = torch.zeros(nProj, dtype=torch.float64)

	x_offset = pGeo[18]
	y_offset = pGeo[19]

	pTubeAngle = torch.linspace(-tubeAngle/2, tubeAngle/2, nProj)
	pDetAngle = torch.linspace(-detAngle/2, detAngle/2, nProj)

	projectionDDb(pProj, pVolume, pTubeAngle, pDetAngle, nProj, nPixX, nPixY, nSlices, nDetX, nDetY, idXProj, x_offset, y_offset, dx, dy, dz, du, dv, DSD, DDR, DAG);
