/*
 * MIT License
 *
 * Copyright (c) 2023 Carnegie Mellon University
 *
 * This file is part of RoWild.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "pfilter.h"
#include "rowild_utils.h"
#include <algorithm>
#include <string>
#include <vector>

ParticleFilter::ParticleFilter(EnvMap *_occGrid, int _numParticles,
                               int _subsample) {
    this->occGrid = _occGrid;
    this->numParticles = _numParticles;
    this->subsample = _subsample;
    this->resolution = occGrid->getResolution();

    this->initializeParticles(occGrid, numParticles);
    this->maxDev = this->getDevIdx();
    this->initialNumParticles = _numParticles;

    this->initializeMotionModel();
    this->initializeSensorModel();
    this->initializeResampler();

    this->initialDev = maxDev;
    this->prevDev = INT_MAX;
}

ParticleFilter::~ParticleFilter() { delete this->resamDist; }

void ParticleFilter::initializeParticles(EnvMap *occGrid, int numParticles) {
    // Randomly sample from free space
    auto freeXs = occGrid->getFreeXs();
    auto freeYs = occGrid->getFreeYs();
    assert(freeXs->size() == freeYs->size());
    int numFreeLocs = static_cast<int>(freeXs->size());

    std::default_random_engine intGen;
    std::uniform_int_distribution<int> intDist(0, numFreeLocs - 1);

    std::default_random_engine realGen;
    std::uniform_real_distribution<double> realDist(-PI, PI);

    for (int i = 0; i < numParticles; i++) {
        int idx = intDist(intGen);
        assert(idx >= 0 && idx < numFreeLocs);

        double x = freeXs->at(idx);
        double y = freeYs->at(idx);
        assert(occGrid->isFree(static_cast<int>(x), static_cast<int>(y)));

        x *= occGrid->getResolution();
        y *= occGrid->getResolution();

        double theta = realDist(realGen);
        assert(theta >= -PI && theta < PI);

        double weight = 1.0 / numParticles;
        this->particles.push_back(Particle(x, y, theta, weight));
    }
}

void ParticleFilter::initializeMotionModel() {
    // These numbers are tuned for CMU's Wean Hall
    this->alpha1 = 0.001;
    this->alpha2 = 0.001;
    this->alpha3 = 0.1;
    this->alpha4 = 0.8;
}

void ParticleFilter::updateMotion(READING *prevOdometry,
                                  READING *currOdometry) {
    double xDiff = currOdometry->at(0) - prevOdometry->at(0);
    double yDiff = currOdometry->at(1) - prevOdometry->at(1);
    double thetaDiff = currOdometry->at(2) - prevOdometry->at(2);

    if (unlikely(std::abs(xDiff) + std::abs(yDiff) + std::abs(thetaDiff) <
                 1e-10)) {
        // Avoid the overheads if the difference is minor
        return;
    }

    double dRot1 = atan2(yDiff, xDiff) - prevOdometry->at(2);
    double dTrans = sqrt(xDiff * xDiff + yDiff * yDiff);
    double dRot2 = currOdometry->at(2) - prevOdometry->at(2) - dRot1;

    std::default_random_engine generator;

    double scaleH1 =
        sqrt(this->alpha1 * dRot1 * dRot1 + this->alpha2 * dTrans * dTrans);

    double scaleTh =
        sqrt(this->alpha3 * dTrans * dTrans + this->alpha4 * dRot1 * dRot1 +
             this->alpha4 * dRot2 * dRot2);

    double scaleH2 =
        sqrt(this->alpha1 * dRot2 * dRot2 + this->alpha2 * dTrans * dTrans);

    std::normal_distribution<double> h1Dist(0, scaleH1);
    std::normal_distribution<double> thDist(0, scaleTh);
    std::normal_distribution<double> h2Dist(0, scaleH2);

    double dRh1 = dRot1 - h1Dist(generator);
    double dTh = dTrans - thDist(generator);
    double dRh2 = dRot2 - h2Dist(generator);

#pragma omp parallel for num_threads(8)
    for (int i = 0; i < this->numParticles; i++) {
        double thetaPrime = this->particles[i].theta + dRh1;

        this->particles[i].x += dTh * cos(thetaPrime);
        this->particles[i].y += dTh * sin(thetaPrime);
        this->particles[i].theta =
            wrapToPi(this->particles[i].theta + dRh1 + dRh2);
    }
}

void ParticleFilter::initializeSensorModel() {
    // These numbers are tuned for CMU's Wean Hall
    this->zHit = 10;
    this->zShort = 0.01;
    this->zMax = 0.1;
    this->zRand = 10;
    this->sigmaHit = 50.0;
    this->lambdaShort = 0.1;

    this->minProbability = 0.35;
    this->maxRange = 1000.0;

    // The distance of the laser from the robot
    this->sensorOffset = 25.0;
}



double ParticleFilter::rayCast(double x, double y, double theta,
                               double degree) const {
    double xRay = x + this->sensorOffset * cos(theta);
    double yRay = y + this->sensorOffset * sin(theta);

    double step = this->resolution;
    double xStep = step * cos(PI / 2 + theta - degToRad(degree));
    double yStep = step * sin(PI / 2 + theta - degToRad(degree));

    double dist = 0;
    while (true) {
        dist += step;
        xRay += xStep;
        yRay += yStep;

        int xIdx = static_cast<int>(xRay / this->resolution);
        int yIdx = static_cast<int>(yRay / this->resolution);

        if (dist >= this->maxRange || xIdx >= this->occGrid->getX() ||
            yIdx >= this->occGrid->getY() || xIdx < 0 || yIdx < 0) {
            break;
        }

        double occ = this->occGrid->getProb(xIdx, yIdx);
        if (occ == -1 || occ >= this->minProbability) {
            break;
        }
    }
    return dist;
}

void ParticleFilter::initializeResampler() {
    assert(this->numParticles > 0);
    this->resamDist =
        new std::uniform_real_distribution<double>(0, 1.0 / this->numParticles);
}

void ParticleFilter::resample() {
    // Low-variance sampling

    double wSum = 0;
    for (int i = 0; i < numParticles; i++) {
        wSum += this->particles[i].w;
    }
    for (int i = 0; i < numParticles; i++) {
        this->particles[i].w /= wSum;
    }

    std::vector<Particle> newParticles;

    double M1 = 1.0 / this->numParticles;
    double r = (*this->resamDist)(this->resamRandGen);
    double c = this->particles[0].w;
    int i = 0;

    for (int m = 0; m < this->numParticles; m++) {
        double u = r + m * M1;
        while (u > c) {
            i++;
            c += this->particles[i].w;
        }
        newParticles.push_back(this->particles[i]);
    }

    this->particles.clear();
    std::copy(newParticles.begin(), newParticles.end(),
              std::back_inserter(this->particles));
}

void ParticleFilter::printOverallStats(std::string header) const {
    // This function can be greatly optimized, but it's off the ROI

    double xMean = 0, yMean = 0, thetaMean = 0, wMean = 0;
    for (int i = 0; i < this->numParticles; i++) {
        Particle p = this->particles[i];

        xMean += p.x;
        yMean += p.y;
        thetaMean += p.theta;
        wMean += p.w;
    }

    xMean /= this->numParticles;
    yMean /= this->numParticles;
    thetaMean /= this->numParticles;
    wMean /= this->numParticles;

    double xDev = 0, yDev = 0, thetaDev = 0, wDev = 0;
    for (int i = 0; i < this->numParticles; i++) {
        Particle p = this->particles[i];

        xDev += (p.x - xMean) * (p.x - xMean);
        yDev += (p.y - yMean) * (p.y - yMean);
        thetaDev += (p.theta - thetaMean) * (p.theta - thetaMean);
        wDev += (p.w - wMean) * (p.w - wMean);
    }

    xDev = sqrt(xDev);
    xDev /= this->numParticles;
    yDev = sqrt(yDev);
    yDev /= this->numParticles;
    thetaDev = sqrt(thetaDev);
    thetaDev /= this->numParticles;
    wDev = sqrt(wDev);
    wDev /= this->numParticles;

    if (header != "") printf("%s:\n", header.c_str());
    printf("Mean: %.2f %.2f %.2f %.2f\n", xMean, yMean, thetaMean, wMean);
    printf("Dev: %.2f %.2f %.2f %.2f\n", xDev, yDev, thetaDev, wDev);
    printf("Num kidnaps: %d\n\n", this->numKidnaps);
    printf("Num reconfigurations: %d\n\n", this->numReconfigurations);
}

void ParticleFilter::printAllParticles(std::string header) const {
    printf("%s:\n", header.c_str());
    for (int i = 0; i < this->numParticles; i++) {
        auto p = this->particles[i];
        printf("Particle %d: X:%.2f, Y:%.2f, Theta:%.2f, W:%.2f\n", i, p.x, p.y,
               p.theta, p.w);
    }
}

std::vector<double> ParticleFilter::getBelief() const {
    // It may look that storing <x, y, theta>s in different vectors and
    // traversing each vector individually would be faster if the compiler can
    // generate AVX instructions out of it. But experimentally I found this
    // implementation somewhat faster. The reason is the significant cache
    // locality, achieved by putting <x, y, theta>s nearby
    double xMean = 0, yMean = 0, thetaMean = 0;
    for (int i = 0; i < this->numParticles; i++) {
        Particle p = this->particles[i];
        xMean += p.x;
        yMean += p.y;
        thetaMean += p.theta;
    }

    xMean /= this->numParticles;
    yMean /= this->numParticles;
    thetaMean /= this->numParticles;

    return {xMean, yMean, thetaMean};
}

double ParticleFilter::getDevIdx() const {
    double xMean = 0, yMean = 0;
    for (int i = 0; i < this->numParticles; i++) {
        Particle p = this->particles[i];
        xMean += p.x;
        yMean += p.y;
    }

    xMean /= this->numParticles;
    yMean /= this->numParticles;

    double xDev = 0, yDev = 0;
    for (int i = 0; i < this->numParticles; i++) {
        Particle p = this->particles[i];
        xDev += (p.x - xMean) * (p.x - xMean);
        yDev += (p.y - yMean) * (p.y - yMean);
    }

    xDev /= this->numParticles;
    yDev /= this->numParticles;

    return xDev + yDev;
}

void ParticleFilter::handleKidnapping(double threshold) {
    double currDev = this->getDevIdx();
    if (unlikely(currDev - this->prevDev > threshold * this->initialDev)) {
        this->numKidnaps++;
        this->particles.clear();
        this->initializeParticles(this->occGrid, this->numParticles);
        this->initialDev = this->getDevIdx();
        this->prevDev = INT_MAX;
    } else {
        this->prevDev = currDev;
    }
}

void ParticleFilter::adjustParticles(int minParticles, int maxParticles,
                                     int step) {
    double currDev = this->getDevIdx();
    int expectedParticles =
        round((currDev / this->maxDev) * this->initialNumParticles);

    // Unless with crazily dynamic scenes, the expected number of particles
    // must not be that large
    if (unlikely(expectedParticles > maxParticles)) {
        expectedParticles = maxParticles;
    } else if (expectedParticles < minParticles) {
        expectedParticles = minParticles;
    }

    if (this->numParticles < expectedParticles) {
        if (this->numParticles + step > maxParticles) return;

        this->initializeParticles(this->occGrid, step);
        this->numParticles = this->particles.size();
        this->numReconfigurations++;
    } else if (this->numParticles > expectedParticles) {
        if (this->numParticles - step < minParticles) return;

        this->particles.erase(this->particles.begin(),
                              this->particles.begin() + step);
        this->numParticles = this->particles.size();
        this->numReconfigurations++;
    }
}

void ParticleFilter::saveParticleState(int updateIdx,
                                      const std::string &particleDir) const {
    std::string particleFile = particleDir + "/particles_" +
                               std::to_string(updateIdx) + ".txt";
    FILE *f = fopen(particleFile.c_str(), "w");
    if (f == nullptr) {
        fprintf(stderr, "[DeliBot] Failed to open %s for writing\n",
                particleFile.c_str());
        return;
    }

    for (const auto &p : this->particles) {
        fprintf(f, "%.2f %.2f %.2f %.2f\n", p.x, p.y, p.theta, p.w);
    }
    fclose(f);
    printf("[DeliBot] Saved %d particles to %s\n", this->numParticles,
           particleFile.c_str());
}

void ParticleFilter::loadParticleState(int updateIdx,
                                      const std::string &particleDir) {
    std::string particleFile = particleDir + "/particles_" +
                               std::to_string(updateIdx) + ".txt";
    FILE *f = fopen(particleFile.c_str(), "r");
    if (f == nullptr) {
        fprintf(stderr, "[DeliBot] Failed to open %s for reading\n",
                particleFile.c_str());
        return;
    }

    this->particles.clear();
    double x, y, theta, w;
    while (fscanf(f, "%lf %lf %lf %lf", &x, &y, &theta, &w) == 4) {
        this->particles.emplace_back(x, y, theta, w);
    }
    fclose(f);
    this->numParticles = this->particles.size();
    if (this->numParticles == 0) {
        fprintf(stderr, "[DeliBot] No particles loaded from %s\n",
                particleFile.c_str());
    } else {
        printf("[DeliBot] Loaded %d particles from %s\n", this->numParticles,
               particleFile.c_str());
    }

}
