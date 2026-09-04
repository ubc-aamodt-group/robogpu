#include "pfilter.h"


#define PI (3.141592654)
#define TWO_PI (6.283185308)


#define MAX_NODES 10000000

#ifdef TTA

__device__ __noinline__ void __TreeUnitDebug(void* value) {
    printf("PLEASE RE-RUN IN GPGPUSIM\n");
    printf("Debug: %d\n", *(int*)value);
    return;
}

__device__ __noinline__ void __TreeUnitWrite(void* dest, void* src, unsigned size) {
    printf("PLEASE RE-RUN IN GPGPUSIM\n");
    printf("Write: %d\n", *(int*)src);
    printf("Write: %d\n", *(int*)dest);
    printf("Size: %d\n", size);
    return;
}

__device__ __noinline__ void __TreeUnitSearch(
	TreeSearchConditions conditions,
	void* rootNode,
	volatile TreeSearchResults* results,
	unsigned node_config,
	unsigned tri_config
)
{
	printf("PLEASE RE-RUN IN GPGPUSIM\n");

    // Use all the variables so they're not optimized out
    printf("Search conditions: ");
    for (int i=0; i<12; i++) {
        printf("%d ", conditions.values[i]);
    }
    printf("\n");

    printf("Root node address: 0x%x\n", rootNode);

    printf("Search results: ");
    for (int i=0; i<8; i++) {
        printf("%d ", results->values[i]);
    }
    printf("\n");

    printf("Node intersection type: %d\n", node_config);
    printf("Leaf intersection type: %d\n", tri_config);

	return;
}

__device__ __noinline__ void __TreeUnitCompleteSearch() {
	printf("Magic Function\n");
	return;
}

#endif

__global__ void calcProbability(Particle *particles, int numParticles, double *laserReading, int numRays, double resolution, int subsample, double *probs, double *occGrid, int mapSizeX, int mapSizeY) {
    int rayIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (rayIdx >= numRays) return;

    int raysPerParticle = 180 / subsample;
    int particleIdx = rayIdx / raysPerParticle;
    int degree = rayIdx % raysPerParticle * subsample;

    Particle *particle = &particles[particleIdx];
    double sensorOffset = 25.0;

    double x = particle->x;
    double y = particle->y;
    double theta = particle->theta;

    double xRay = x + sensorOffset * cos(theta);
    double yRay = y + sensorOffset * sin(theta);

    double step = resolution;
    double dist = 0;

    double xStep = step * cos(PI / 2 + theta - degree * (PI / 180));
    double yStep = step * sin(PI / 2 + theta - degree * (PI / 180));

    #ifdef TTA
    double probability = 1.0;
    double zkt = laserReading[(int)degree];
    // // Take first step
    // dist += step;
    // xRay += xStep;
    // yRay += yStep;

    TreeSearchConditions conditions;
    uint32_t* base;
    base = (uint32_t *)&xRay;
    conditions.values[0] = base[0];
    conditions.values[1] = base[1];
    base = (uint32_t *)&yRay;
    conditions.values[2] = base[0];
    conditions.values[3] = base[1];
    base = (uint32_t *)&xStep;
    conditions.values[4] = base[0];
    conditions.values[5] = base[1];
    base = (uint32_t *)&yStep;
    conditions.values[6] = base[0];
    conditions.values[7] = base[1];
    base = (uint32_t *)&dist;
    conditions.values[8] = base[0];
    conditions.values[9] = base[1];
    base = (uint32_t *)&resolution;
    conditions.values[10] = base[0];
    conditions.values[11] = base[1];
    base = (uint32_t *)&zkt;
    conditions.values[12] = base[0];
    conditions.values[13] = base[1];


    conditions.values[14] = *(uint32_t *)&mapSizeX;
    conditions.values[15] = *(uint32_t *)&mapSizeY;

    __shared__ volatile TreeSearchResults results[64]; // 64 * sizeof(TreeSearchResults) = 256 * sizeof(double)

    volatile unsigned node_config = 10;
    volatile unsigned leaf_config = 10;
    
    int shmem_tid = threadIdx.x; // 0-255
    volatile TreeSearchResults* resultsPtr = &results[shmem_tid / 4]; // Index to results[64] array
    resultsPtr = (volatile TreeSearchResults*)((void *)resultsPtr + (shmem_tid % 4) * sizeof(double)); // Adjust for offset to (double) type

    __TreeUnitSearch(
        conditions,
        &occGrid[0],
        resultsPtr,
        node_config,
        leaf_config
    );
    __TreeUnitCompleteSearch();

    volatile unsigned byte = sizeof(double);
    probs[rayIdx] = *(double *)&resultsPtr->values[0];

    #else
    double zHit = 10;
    double zShort = 0.01;
    double zMax = 0.1;
    double zRand = 10;
    double sigmaHit = 50.0;
    double lambdaShort = 0.1;
    double minProbability = 0.35;
    double maxRange = 1000.0;

    unsigned iterations = 0;

    // printf("[%3d] Ray-cast: x = %.2f, y = %.2f, xstep = %.2f, ystep = %.2f, theta = %.2f, degree = %.2f\n", rayIdx, xRay, yRay, xStep, yStep, theta, degree);
    while (true) {
        iterations++;
        dist += step;
        xRay += xStep;
        yRay += yStep;

        int xIdx = static_cast<int>(xRay / resolution);
        int yIdx = static_cast<int>(yRay / resolution);

        if (dist >= maxRange || xIdx >= mapSizeX || yIdx >= mapSizeY ||
            xIdx < 0 || yIdx < 0) {
            // printf("[%3d] Ray-cast: Out of bounds.\n", rayIdx);
            break;
        }

        int gIdx = xIdx * mapSizeY + yIdx;
        double occ = occGrid[gIdx];
        // printf("[%3d] Ray-cast: xIdx = %d, yIdx = %d, gIdx = %d, occ = %.2f\n", rayIdx, xIdx, yIdx, gIdx, occ);
        if (occ == -1 || occ >= minProbability) {
            // printf("[%3d] Ray-cast: Hit an obstacle.\n", rayIdx);
            break;
        }
    }
    // printf("[%3d] Ray-cast: x = %.2f, y = %.2f, theta = %.2f, degree = %.2f, dist = %.2f\n", rayIdx, xRay, yRay, theta, degree, dist);

    double zkt = laserReading[(int)degree];
    double zktStar = dist;
    double pRand = (zkt >= 0 && zkt < maxRange) ? 1.0 / maxRange : 0.0;
    double pMax = (zkt >= maxRange) ? 1.0 : 0.0;
    double pShort = (zkt >= 0 && zkt <= zktStar)
                        ? (1.0 / (1 - exp(-lambdaShort * zktStar))) *
                            lambdaShort * exp(-lambdaShort * zkt)
                        : 0.0;
    double pHit = (zkt >= 0 && zkt <= maxRange)
                    ? exp(-0.5 * (zkt - zktStar) * (zkt - zktStar) /
                            (sigmaHit * sigmaHit)) /
                            sqrt(TWO_PI * sigmaHit * sigmaHit)
                    : 0.0;
    double probability =
        zHit * pHit + zShort * pShort + zMax * pMax + zRand * pRand;

    probs[rayIdx] = probability;
    #endif
    // printf("Particle %d: x = %.2f, y = %.2f, theta = %.2f, prob = %.2f\n", particleIdx, x, y, theta, probability);

}

void ParticleFilter::updateSensor(READING *laserReading) {
    int numParticles = this->numParticles;
    int raysPerParticle = 180 / this->subsample;
    int numRays = numParticles * raysPerParticle;

    // Copy particles to device
    Particle *d_particles;
    Particle *h_particles;
    h_particles = (Particle *)malloc(numParticles * sizeof(Particle));
    for (int i = 0; i < numParticles; i++) {
        h_particles[i] = this->particles[i];
    }
    cudaMalloc((void **)&d_particles, numParticles * sizeof(Particle));
    cudaMemcpy(d_particles, h_particles, numParticles * sizeof(Particle), cudaMemcpyHostToDevice);

    // Allocate memory of probability results
    double *d_probs;
    cudaMalloc((void **)&d_probs, numRays * sizeof(double));

    // Allocate memory for laser reading
    double *d_laserReading;
    int laserReadingSize = laserReading->size();
    double* h_laserReading = new double[laserReadingSize];
    for (int i = 0; i < laserReadingSize; i++) {
        h_laserReading[i] = laserReading->at(i);
    }
    cudaMalloc((void **)&d_laserReading, laserReadingSize * sizeof(double));
    cudaMemcpy(d_laserReading, h_laserReading, laserReadingSize * sizeof(double), cudaMemcpyHostToDevice);

    // Allocate memory for occupancy grid
    double *d_occGrid;
    double *h_occGrid = new double[this->occGrid->getX() * this->occGrid->getY()];
    for (int i = 0; i < this->occGrid->getX(); i++) {
        for (int j = 0; j < this->occGrid->getY(); j++) {
            h_occGrid[i * this->occGrid->getY() + j] = this->occGrid->getProb(i, j);
        }
    }

    // printf("\n\n========================\n");
    // printf("Occupancy grid[61][411]: %.2f\n", h_occGrid[61 * this->occGrid->getY() + 411]);
    // printf("Occupancy grid[232][403]: %.2f\n", h_occGrid[232 * this->occGrid->getY() + 403]);


    cudaMalloc((void **)&d_occGrid, this->occGrid->getX() * this->occGrid->getY() * sizeof(double));
    cudaMemcpy(d_occGrid, h_occGrid, this->occGrid->getX() * this->occGrid->getY() * sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (numRays + blockSize - 1) / blockSize;
    calcProbability<<<numBlocks, blockSize>>>(d_particles, numParticles, d_laserReading, numRays, this->resolution, this->subsample, d_probs, d_occGrid, this->occGrid->getX(), this->occGrid->getY());

    cudaDeviceSynchronize();
    // Check for errors 
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }

    // Copy results back to host
    double *h_probs = new double[numRays];
    cudaMemcpy(h_probs, d_probs, numRays * sizeof(double), cudaMemcpyDeviceToHost);

    // Update particles (reduce)
    for (int i = 0; i < numParticles; i++) {
        double probability = 1.0;
        for (int j = 0; j < raysPerParticle; j++) {
            int rayIdx = i * raysPerParticle + j;
            // printf("Particle %d: x = %.2f, y = %.2f, theta = %.2f, prob[%d] = %.2f\n", i, this->particles[i].x, this->particles[i].y, this->particles[i].theta, j, h_probs[rayIdx]);
            probability *= h_probs[rayIdx];
        }
        this->particles[i].w = probability;
        // printf("Particle %d: x = %.2f, y = %.2f, theta = %.2f, prob[final] = %.2f\n", i, this->particles[i].x, this->particles[i].y, this->particles[i].theta, probability);
    }

    // Free memory
    cudaFree(d_particles);
    cudaFree(d_probs);
    cudaFree(d_laserReading);
    cudaFree(d_occGrid);
    free(h_particles);
    delete[] h_laserReading;
    delete[] h_probs;
    delete[] h_occGrid;
}
