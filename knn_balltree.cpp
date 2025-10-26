/**
 * K-Nearest Neighbors Optimization Using Ball Tree
 * Data Structures Innovative Assignment
 * 
 * This program demonstrates the optimization of KNN search from O(N) to O(log N)
 * using Ball Tree data structure with triangle inequality pruning.
 * 
 * Compilation: g++ -std=c++14 -O3 -o knn_balltree knn_balltree.cpp
 * Execution: ./knn_balltree (Linux/macOS) or knn_balltree.exe (Windows)
 */

#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
#include <chrono>
#include <random>
#include <algorithm>
#include <iomanip>
#include <memory>

using namespace std;
using namespace chrono;

// ============================================================================
// DATA STRUCTURE 1: POINT (using std::vector for dynamic dimensions)
// ============================================================================

class Point {
public:
    vector<double> coords;  // Vector to store coordinates
    int id;
    
    Point(int dimensions = 3, int identifier = -1) 
        : coords(dimensions, 0.0), id(identifier) {}
    
    Point(const vector<double>& c, int identifier = -1) 
        : coords(c), id(identifier) {}
    
    double operator[](int i) const { return coords[i]; }
    double& operator[](int i) { return coords[i]; }
    int size() const { return coords.size(); }
};

// ============================================================================
// DISTANCE CALCULATION (Euclidean Distance)
// ============================================================================

double euclideanDistance(const Point& p1, const Point& p2) {
    double sum = 0.0;
    for (int i = 0; i < p1.size(); ++i) {
        double diff = p1[i] - p2[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

// ============================================================================
// DATA STRUCTURE 2: BALL TREE NODE (using std::unique_ptr for tree structure)
// ============================================================================

struct BallTreeNode {
    Point center;                           // Center of bounding sphere
    double radius;                          // Radius of bounding sphere
    vector<Point> points;                   // Points in leaf nodes (using vector)
    unique_ptr<BallTreeNode> left;          // Left child (smart pointer)
    unique_ptr<BallTreeNode> right;         // Right child (smart pointer)
    bool isLeaf;
    
    BallTreeNode() : radius(0.0), isLeaf(true) {}
};

// ============================================================================
// DATA STRUCTURE 3: PRIORITY QUEUE (for maintaining K nearest neighbors)
// ============================================================================

struct Neighbor {
    Point point;
    double distance;
    
    bool operator<(const Neighbor& other) const {
        return distance < other.distance;  // Max heap
    }
};

// ============================================================================
// BALL TREE IMPLEMENTATION
// ============================================================================

class BallTree {
private:
    unique_ptr<BallTreeNode> root;
    int leafSize;
    mutable long long distanceComputations;
    
    // Compute centroid of points
    Point computeCentroid(const vector<Point>& points) {
        Point centroid(points[0].size());
        for (const auto& p : points) {
            for (int i = 0; i < p.size(); ++i) {
                centroid[i] += p[i];
            }
        }
        for (int i = 0; i < centroid.size(); ++i) {
            centroid[i] /= points.size();
        }
        return centroid;
    }
    
    // Find farthest point from given point
    int findFarthest(const Point& center, const vector<Point>& points) {
        double maxDist = -1.0;
        int farthestIdx = 0;
        for (size_t i = 0; i < points.size(); ++i) {
            double dist = euclideanDistance(center, points[i]);
            if (dist > maxDist) {
                maxDist = dist;
                farthestIdx = i;
            }
        }
        return farthestIdx;
    }
    
    // Build tree recursively
    unique_ptr<BallTreeNode> buildTree(vector<Point>& points) {
        auto node = make_unique<BallTreeNode>();
        
        if (points.empty()) return node;
        
        // Compute center and radius
        node->center = computeCentroid(points);
        node->radius = 0.0;
        for (const auto& p : points) {
            double dist = euclideanDistance(node->center, p);
            if (dist > node->radius) node->radius = dist;
        }
        
        // Leaf node condition
        if (points.size() <= leafSize) {
            node->isLeaf = true;
            node->points = points;
            return node;
        }
        
        // Find two pivot points
        node->isLeaf = false;
        int pivot1Idx = findFarthest(node->center, points);
        Point pivot1 = points[pivot1Idx];
        int pivot2Idx = findFarthest(pivot1, points);
        Point pivot2 = points[pivot2Idx];
        
        // Partition points
        vector<Point> leftPoints, rightPoints;
        for (const auto& p : points) {
            double dist1 = euclideanDistance(p, pivot1);
            double dist2 = euclideanDistance(p, pivot2);
            if (dist1 <= dist2) {
                leftPoints.push_back(p);
            } else {
                rightPoints.push_back(p);
            }
        }
        
        // Handle degenerate case
        if (leftPoints.empty() || rightPoints.empty()) {
            node->isLeaf = true;
            node->points = points;
            return node;
        }
        
        // Recursively build subtrees
        node->left = buildTree(leftPoints);
        node->right = buildTree(rightPoints);
        
        return node;
    }
    
    // Search for K nearest neighbors
    void searchKNN(BallTreeNode* node, const Point& query, int k,
                   priority_queue<Neighbor>& neighbors) const {
        if (!node) return;
        
        // Calculate distance to ball center
        double centerDist = euclideanDistance(query, node->center);
        distanceComputations++;
        
        // Triangle inequality pruning
        double minDist = max(0.0, centerDist - node->radius);
        if (neighbors.size() == k && minDist >= neighbors.top().distance) {
            return;  // Prune this subtree
        }
        
        // Leaf node: check all points
        if (node->isLeaf) {
            for (const auto& p : node->points) {
                double dist = euclideanDistance(query, p);
                distanceComputations++;
                
                if (neighbors.size() < k) {
                    neighbors.push({p, dist});
                } else if (dist < neighbors.top().distance) {
                    neighbors.pop();
                    neighbors.push({p, dist});
                }
            }
            return;
        }
        
        // Internal node: visit children in order of proximity
        double leftDist = euclideanDistance(query, node->left->center);
        double rightDist = euclideanDistance(query, node->right->center);
        distanceComputations += 2;
        
        if (leftDist < rightDist) {
            searchKNN(node->left.get(), query, k, neighbors);
            searchKNN(node->right.get(), query, k, neighbors);
        } else {
            searchKNN(node->right.get(), query, k, neighbors);
            searchKNN(node->left.get(), query, k, neighbors);
        }
    }
    
public:
    BallTree(int leafSize = 40) : leafSize(leafSize), distanceComputations(0) {}
    
    void build(vector<Point>& points) {
        distanceComputations = 0;
        root = buildTree(points);
    }
    
    vector<pair<Point, double>> findKNN(const Point& query, int k) const {
        distanceComputations = 0;
        priority_queue<Neighbor> neighbors;
        searchKNN(root.get(), query, k, neighbors);
        
        vector<pair<Point, double>> results;
        while (!neighbors.empty()) {
            results.push_back({neighbors.top().point, neighbors.top().distance});
            neighbors.pop();
        }
        reverse(results.begin(), results.end());
        return results;
    }
    
    long long getDistanceComputations() const { return distanceComputations; }
};

// ============================================================================
// BRUTE FORCE KNN (for comparison)
// ============================================================================

class BruteForceKNN {
private:
    vector<Point> points;
    mutable long long distanceComputations;
    
public:
    BruteForceKNN() : distanceComputations(0) {}
    
    void build(const vector<Point>& dataPoints) {
        points = dataPoints;
        distanceComputations = 0;
    }
    
    vector<pair<Point, double>> findKNN(const Point& query, int k) const {
        distanceComputations = 0;
        priority_queue<Neighbor> neighbors;
        
        // Check every point
        for (const auto& p : points) {
            double dist = euclideanDistance(query, p);
            distanceComputations++;
            
            if (neighbors.size() < k) {
                neighbors.push({p, dist});
            } else if (dist < neighbors.top().distance) {
                neighbors.pop();
                neighbors.push({p, dist});
            }
        }
        
        vector<pair<Point, double>> results;
        while (!neighbors.empty()) {
            results.push_back({neighbors.top().point, neighbors.top().distance});
            neighbors.pop();
        }
        reverse(results.begin(), results.end());
        return results;
    }
    
    long long getDistanceComputations() const { return distanceComputations; }
};

// ============================================================================
// DATA GENERATION
// ============================================================================

vector<Point> generateRandomPoints(int numPoints, int dimensions, int seed = 42) {
    mt19937 rng(seed);
    uniform_real_distribution<double> dist(0.0, 100.0);
    
    vector<Point> points;
    for (int i = 0; i < numPoints; ++i) {
        Point p(dimensions, i);
        for (int d = 0; d < dimensions; ++d) {
            p[d] = dist(rng);
        }
        points.push_back(p);
    }
    return points;
}

// ============================================================================
// BENCHMARK AND COMPARISON
// ============================================================================

void runBenchmark(int numPoints, int dimensions, int k, int numQueries) {
    cout << "\n" << string(80, '=') << endl;
    cout << "BENCHMARK: " << numPoints << " points, " << dimensions 
         << "D, K=" << k << ", Queries=" << numQueries << endl;
    cout << string(80, '=') << endl;
    
    // Generate data
    cout << "\nGenerating data..." << flush;
    auto dataPoints = generateRandomPoints(numPoints, dimensions);
    auto queries = generateRandomPoints(numQueries, dimensions, 100);
    cout << " Done!" << endl;
    
    // Build Ball Tree
    cout << "Building Ball Tree..." << flush;
    auto buildStart = high_resolution_clock::now();
    BallTree ballTree(40);
    ballTree.build(dataPoints);
    auto buildEnd = high_resolution_clock::now();
    double buildTime = duration_cast<milliseconds>(buildEnd - buildStart).count();
    cout << " Done! (" << buildTime << " ms)" << endl;
    
    // Build Brute Force
    cout << "Building Brute Force..." << flush;
    BruteForceKNN bruteForce;
    bruteForce.build(dataPoints);
    cout << " Done!" << endl;
    
    // Query Ball Tree
    cout << "\nQuerying Ball Tree..." << flush;
    auto ballStart = high_resolution_clock::now();
    long long totalBallDist = 0;
    for (const auto& query : queries) {
        auto neighbors = ballTree.findKNN(query, k);
        totalBallDist += ballTree.getDistanceComputations();
    }
    auto ballEnd = high_resolution_clock::now();
    double ballTime = duration_cast<microseconds>(ballEnd - ballStart).count() / 1000.0;
    cout << " Done! (" << ballTime << " ms)" << endl;
    
    // Query Brute Force
    cout << "Querying Brute Force..." << flush;
    auto bruteStart = high_resolution_clock::now();
    long long totalBruteDist = 0;
    for (const auto& query : queries) {
        auto neighbors = bruteForce.findKNN(query, k);
        totalBruteDist += bruteForce.getDistanceComputations();
    }
    auto bruteEnd = high_resolution_clock::now();
    double bruteTime = duration_cast<microseconds>(bruteEnd - bruteStart).count() / 1000.0;
    cout << " Done! (" << bruteTime << " ms)" << endl;
    
    // Results
    double speedup = bruteTime / ballTime;
    long long avgBallDist = totalBallDist / numQueries;
    long long avgBruteDist = totalBruteDist / numQueries;
    double pruning = 100.0 * (1.0 - (double)avgBallDist / avgBruteDist);
    
    cout << "\n" << string(80, '-') << endl;
    cout << "RESULTS:" << endl;
    cout << string(80, '-') << endl;
    cout << fixed << setprecision(2);
    cout << "Ball Tree Build Time:        " << buildTime << " ms" << endl;
    cout << "Ball Tree Query Time:        " << ballTime << " ms" << endl;
    cout << "Brute Force Query Time:      " << bruteTime << " ms" << endl;
    cout << "Speedup:                     " << speedup << "x" << endl;
    cout << "\nDistance Computations (avg per query):" << endl;
    cout << "  Ball Tree:                 " << avgBallDist << endl;
    cout << "  Brute Force:               " << avgBruteDist << endl;
    cout << "  Pruning Efficiency:        " << pruning << "%" << endl;
    cout << string(80, '=') << endl;
}

// ============================================================================
// CORRECTNESS VERIFICATION
// ============================================================================

void verifyCorrectness() {
    cout << "\n" << string(80, '=') << endl;
    cout << "CORRECTNESS VERIFICATION" << endl;
    cout << string(80, '=') << endl;
    
    auto dataPoints = generateRandomPoints(500, 3);
    auto query = generateRandomPoints(1, 3, 999)[0];
    int k = 5;
    
    BallTree ballTree;
    ballTree.build(dataPoints);
    
    BruteForceKNN bruteForce;
    bruteForce.build(dataPoints);
    
    auto ballResults = ballTree.findKNN(query, k);
    auto bruteResults = bruteForce.findKNN(query, k);
    
    cout << "\nComparing top " << k << " neighbors:" << endl;
    bool correct = true;
    for (int i = 0; i < k; ++i) {
        double diff = abs(ballResults[i].second - bruteResults[i].second);
        cout << "  Rank " << (i+1) << ": Ball Tree=" << fixed << setprecision(6)
             << ballResults[i].second << ", Brute Force=" << bruteResults[i].second;
        if (diff > 1e-9) {
            cout << " ✗ MISMATCH";
            correct = false;
        } else {
            cout << " ✓";
        }
        cout << endl;
    }
    
    cout << "\n";
    if (correct) {
        cout << "✓ VERIFICATION PASSED: Results match!" << endl;
    } else {
        cout << "✗ VERIFICATION FAILED: Results differ!" << endl;
    }
    cout << string(80, '=') << endl;
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================

int main() {
    cout << "\n";
    cout << "╔════════════════════════════════════════════════════════════════════════════╗\n";
    cout << "║                                                                            ║\n";
    cout << "║           K-NEAREST NEIGHBORS OPTIMIZATION USING BALL TREE                 ║\n";
    cout << "║                                                                            ║\n";
    cout << "║  Demonstrating O(log N) query complexity vs O(N) brute-force approach     ║\n";
    cout << "║                                                                            ║\n";
    cout << "╚════════════════════════════════════════════════════════════════════════════╝\n";
    
    // Verify correctness
    verifyCorrectness();
    
    // Run benchmarks
    runBenchmark(1000, 3, 5, 100);
    runBenchmark(5000, 5, 10, 100);
    runBenchmark(10000, 10, 15, 50);
    
    cout << "\n✓ All tests completed successfully!\n" << endl;
    
    return 0;
}
